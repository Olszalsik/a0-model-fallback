"""Tests for the two guards merged into model_fallback (v3.3.0).

Covers:
  * asyncio read-ready guard -- installs, no-ops a None callback, forwards
    a real callback, is idempotent, and detects an ALREADY-FIXED upstream
    (the case the old ``co_names``-length heuristic got wrong).
  * extension import guard -- a broken file is skipped while its healthy
    siblings still load, the loader is restored afterwards, recovery is
    logged, and the import_module swap does not leak across threads.
"""
from __future__ import annotations

import asyncio.selector_events as se
import os
import sys
import textwrap
import threading

import pytest

REPO_ROOT = os.environ.get("REPO_ROOT_OVERRIDE") or os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from helpers import modules  # noqa: E402
from helpers.extension import Extension  # noqa: E402
from usr.plugins.model_fallback.helpers import asyncio_guard  # noqa: E402
from usr.plugins.model_fallback.helpers import import_guard  # noqa: E402
from usr.plugins.model_fallback.helpers import toggles  # noqa: E402

# --------------------------------------------------------------------------
# asyncio read-ready guard
# --------------------------------------------------------------------------


def test_guard_installs_and_reports_applied():
    assert asyncio_guard.install() is True
    assert asyncio_guard.is_applied() is True


def test_guard_is_idempotent():
    assert asyncio_guard.install() is True
    first = se._SelectorSocketTransport._read_ready
    assert asyncio_guard.install() is True
    assert se._SelectorSocketTransport._read_ready is first


def test_read_ready_noops_when_cb_is_none():
    """The gh-115514 doom-loop: must not raise TypeError."""
    asyncio_guard.install()
    cls = se._SelectorSocketTransport
    inst = cls.__new__(cls)
    inst._read_ready_cb = None
    cls._read_ready(inst)  # would raise TypeError without the guard


def test_read_ready_forwards_when_cb_is_set():
    asyncio_guard.install()
    cls = se._SelectorSocketTransport
    inst = cls.__new__(cls)
    calls = []
    inst._read_ready_cb = lambda: calls.append(1)
    cls._read_ready(inst)
    assert calls == [1]


def test_already_guarded_upstream_is_detected():
    """A fixed CPython must NOT be re-patched.

    Regression: the standalone plugin decided "already guarded" by
    ``len(code.co_names) > 1``. A fixed body is
    ``if cb is None: return`` + call, whose co_names is still exactly
    ``('_read_ready_cb',)`` -- identical to the unguarded body, so the
    old heuristic could never tell them apart.
    """
    class Fixed:
        def _read_ready(self):
            if self._read_ready_cb is None:
                return
            self._read_ready_cb()

    class Unguarded:
        def _read_ready(self):
            self._read_ready_cb()

    assert (
        Fixed._read_ready.__code__.co_names
        == Unguarded._read_ready.__code__.co_names
    ), "premise: the co_names heuristic genuinely cannot distinguish these"

    assert asyncio_guard._needs_guard(Fixed) is False
    assert asyncio_guard._needs_guard(Unguarded) is True


def test_already_guarded_is_not_patched_end_to_end(monkeypatch):
    """install() must leave an already-fixed interpreter untouched."""
    import types

    class Fake:
        def _read_ready(self):
            if self._read_ready_cb is None:
                return
            self._read_ready_cb()

    fake = types.ModuleType("fake_selector_events")
    fake._SelectorSocketTransport = Fake
    monkeypatch.setattr(asyncio_guard, "_selector_events", fake, raising=False)

    original = Fake.__dict__["_read_ready"]
    assert asyncio_guard.install() is True
    assert Fake.__dict__["_read_ready"] is original, (
        "must not overwrite a guarded body"
    )
    assert getattr(Fake, asyncio_guard.PATCHED_FLAG, False) is True


def test_uninstall_is_a_documented_noop():
    assert asyncio_guard.uninstall() is False
    assert asyncio_guard.is_applied() is True


def test_status_shape():
    status = asyncio_guard.status()
    assert set(status) == {"installed", "transport_present"}
    assert status["transport_present"] is True

# --------------------------------------------------------------------------
# extension import guard
# --------------------------------------------------------------------------


def _write_ext(folder, name, body):
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(body))
    return path


_GOOD = """
    from helpers.extension import Extension

    class {name}(Extension):
        def execute(self, **kwargs):
            return None
"""


@pytest.fixture
def guarded_loader():
    """Install the guard; always restore the pristine upstream loader."""
    pristine = modules.load_classes_from_folder
    import_guard.install()
    yield
    modules.load_classes_from_folder = pristine
    import_guard.SKIPPED.clear()
    import_guard._depth = 0
    import_guard._original_import = None


def test_broken_file_is_skipped_and_siblings_load(tmp_path, guarded_loader):
    _write_ext(tmp_path, "_10_good.py", _GOOD.format(name="Good"))
    _write_ext(
        tmp_path,
        "_20_broken.py",
        """
        from helpers.extension import Extension
        from totally_missing_symbol_xyz import boom  # noqa: F401

        class Broken(Extension):
            def execute(self, **kwargs):
                return None
        """,
    )
    _write_ext(tmp_path, "_30_also_good.py", _GOOD.format(name="AlsoGood"))

    classes = modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    names = {c.__name__ for c in classes}

    # The whole point: upstream raises here instead of returning.
    assert "Good" in names and "AlsoGood" in names
    assert "Broken" not in names
    assert len(import_guard.SKIPPED) == 1
    broken_path = next(iter(import_guard.SKIPPED))
    assert broken_path.endswith("_20_broken.py")
    assert "totally_missing_symbol_xyz" in import_guard.SKIPPED[broken_path]


def test_unpatched_loader_still_raises(tmp_path):
    """Control: proves the test above is meaningful, not vacuous."""
    _write_ext(tmp_path, "_20_broken.py", "import totally_missing_symbol_xyz\n")
    with pytest.raises(ImportError):
        modules.load_classes_from_folder(str(tmp_path), "*", Extension)


def test_import_module_restored_after_normal_load(tmp_path, guarded_loader):
    _write_ext(tmp_path, "_10_ok.py", _GOOD.format(name="Ok"))
    pristine_import = modules.import_module
    modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    assert modules.import_module is pristine_import


def test_import_module_restored_even_when_a_file_raises(tmp_path, guarded_loader):
    _write_ext(tmp_path, "_20_broken.py", "import totally_missing_symbol_xyz\n")
    pristine_import = modules.import_module
    modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    assert modules.import_module is pristine_import


def test_recovery_is_logged_and_clears_the_skip(tmp_path, guarded_loader):
    path = _write_ext(tmp_path, "_20_flaky.py", "import totally_missing_symbol_xyz\n")
    modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    assert path in import_guard.SKIPPED

    _write_ext(tmp_path, "_20_flaky.py", _GOOD.format(name="Flaky"))
    classes = modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    assert "Flaky" in {c.__name__ for c in classes}
    assert path not in import_guard.SKIPPED


def test_base_class_filtering_is_preserved(tmp_path, guarded_loader):
    """A skipped file must not smuggle a foreign class into the result."""
    _write_ext(tmp_path, "_20_broken.py", "import totally_missing_symbol_xyz\n")
    classes = modules.load_classes_from_folder(str(tmp_path), "*", Extension)
    assert classes == []
    assert all(c is not Extension for c in classes)
    assert all(issubclass(c, Extension) for c in classes)

def test_concurrent_loads_do_not_leak_the_shim(tmp_path, guarded_loader):
    """Regression for the save/restore race.

    Concurrent threads must not leave ``modules.import_module``
    permanently swapped: one thread saving the shim as "the original"
    while another restores it.
    """
    _write_ext(tmp_path, "_10_ok.py", _GOOD.format(name="Ok"))
    _write_ext(tmp_path, "_20_broken.py", "import totally_missing_symbol_xyz\n")

    pristine_import = modules.import_module
    pristine_loader = modules.load_classes_from_folder
    barrier = threading.Barrier(4)
    errors = []

    def worker():
        try:
            barrier.wait(timeout=10)
            modules.load_classes_from_folder(str(tmp_path), "*", Extension)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"concurrent loads raised: {errors}"
    assert modules.import_module is pristine_import
    assert modules.load_classes_from_folder is pristine_loader
    assert import_guard._depth == 0


def test_uninstall_restores_the_upstream_loader(guarded_loader):
    pristine = getattr(
        modules.load_classes_from_folder, "_import_guard_original", None
    )
    assert import_guard.uninstall() is True
    assert modules.load_classes_from_folder is pristine
    assert not getattr(
        modules.load_classes_from_folder, "_import_guard_patched", False
    )
    assert import_guard.uninstall() is False  # idempotent no-op


def test_drift_check_reports_nothing_for_current_upstream():
    assert import_guard.check_upstream_drift() == []


# --------------------------------------------------------------------------
# toggles
# --------------------------------------------------------------------------


def test_new_guard_toggles_resolve():
    assert toggles.resolve_toggle({}, "asyncio_read_ready_guard") is True
    assert toggles.resolve_toggle({}, "extension_import_guard") is True
    assert (
        toggles.resolve_toggle(
            {"asyncio_read_ready_guard_enabled": False},
            "asyncio_read_ready_guard",
        )
        is False
    )
    assert (
        toggles.resolve_toggle(
            {"extension_import_guard_enabled": False}, "extension_import_guard"
        )
        is False
    )
    # nested back-compat key still honoured
    assert (
        toggles.resolve_toggle(
            {"asyncio_read_ready_guard": {"enabled": False}},
            "asyncio_read_ready_guard",
        )
        is False
    )

