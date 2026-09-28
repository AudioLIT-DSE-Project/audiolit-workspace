"""Worker recovery after a crash.

The per-family GPU lock used to be held for 24h without renewal, and the
`worker all` parent never respawned a dead child. An OOM kill during a dataset
warmup therefore left the ASR family dead (lock held, no process), and every
later ASR job - warmups included - sat queued forever. `docker stop` did the
same to every GPU family by SIGKILLing children that were never signalled.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
from fakeredis import FakeServer, FakeStrictRedis

from app.orchestration import task_orchestrator
from app.orchestration.task_orchestrator import (
    WORKER_LOCK_PREFIX,
    WORKER_LOCK_TTL,
    WorkerFamily,
    _acquire_family_lock,
    _FamilyLockRenewer,
)
from app.orchestration.worker import supervise


@pytest.fixture
def conn():
    fake = FakeStrictRedis(server=FakeServer())
    yield fake
    fake.flushall()


def _key(fam: WorkerFamily) -> str:
    return f"{WORKER_LOCK_PREFIX}:{fam.value}"


class TestFamilyLock:
    def test_lock_is_short_lived_not_24h(self, conn):
        _acquire_family_lock(conn, WorkerFamily.ASR, wait=0)
        assert 0 < conn.ttl(_key(WorkerFamily.ASR)) <= WORKER_LOCK_TTL

    def test_waits_for_a_dead_holders_lock_to_lapse(self, conn):
        # A crashed predecessor: its lock is still there but nobody renews it.
        conn.set(_key(WorkerFamily.ASR), "dead-worker-token", px=300)
        lock = _acquire_family_lock(conn, WorkerFamily.ASR, wait=5, poll=0.05)
        assert lock.owned()

    def test_release_only_frees_its_own_lock(self, conn):
        lock = _acquire_family_lock(conn, WorkerFamily.ADD, wait=0)
        conn.set(_key(WorkerFamily.ADD), "someone-else")  # ours lapsed and was retaken
        lock.release()
        assert conn.get(_key(WorkerFamily.ADD)) == b"someone-else"

    def test_gives_up_when_a_live_holder_keeps_it(self, conn):
        conn.set(_key(WorkerFamily.ASR), "live-worker-token", ex=WORKER_LOCK_TTL)
        with pytest.raises(RuntimeError, match="already has a worker running"):
            _acquire_family_lock(conn, WorkerFamily.ASR, wait=0.2, poll=0.05)

    def test_renewer_keeps_the_lock_alive(self, conn):
        lock = _acquire_family_lock(conn, WorkerFamily.SER, wait=0)
        lock.ttl = 1  # shrink the TTL each renewal restores
        conn.pexpire(_key(WorkerFamily.SER), 1000)
        renewer = _FamilyLockRenewer(lock, WorkerFamily.SER, interval=0.2)
        renewer.start()
        try:
            time.sleep(1.5)  # well past the 1 s TTL
            assert lock.owned()
        finally:
            renewer.stop()


# --- supervise() ------------------------------------------------------------
# Children are real processes; each start drops a marker file so the parent
# test can count starts across the process boundary.

_MARKER_DIR_ENV = "AUDIOLIT_TEST_SUPERVISE_DIR"


def _exit_immediately(fam: WorkerFamily) -> None:
    Path(os.environ[_MARKER_DIR_ENV], f"{fam.value}-{uuid.uuid4().hex}").touch()


def _starts(marker_dir: Path, fam: WorkerFamily) -> int:
    return len(list(marker_dir.glob(f"{fam.value}-*")))


def test_supervise_respawns_a_family_whose_process_died(tmp_path, monkeypatch):
    monkeypatch.setenv(_MARKER_DIR_ENV, str(tmp_path))
    launched = time.monotonic()
    first_seen: list[float] = []
    # Hard ceiling so a supervisor that never respawns fails instead of hanging.
    hard_deadline = launched + 180

    def done() -> bool:
        n = _starts(tmp_path, WorkerFamily.ASR)
        if not first_seen and n:
            first_seen.append(time.monotonic())
        if n >= 3:
            return True
        if first_seen:
            # Budget the wait from the observed cost of one child start-up
            # rather than from a fixed wall-clock figure. Under the spawn start
            # method the child re-imports the application, torch included, before
            # it records anything, which measured ~16.7 s per start on Windows
            # against about a second on Linux. A flat 30 s from launch bought
            # exactly one start here, so this asserted that a working supervisor
            # was broken. The cost is measured, then three of them are allowed.
            one_start = first_seen[0] - launched
            return time.monotonic() - first_seen[0] > max(6.0, one_start * 3)
        return time.monotonic() > hard_deadline

    supervise(
        [WorkerFamily.ASR], target=_exit_immediately, interval=0.05, backoff=0, should_stop=done
    )

    assert _starts(tmp_path, WorkerFamily.ASR) >= 3


def test_supervise_backs_off_a_crash_looping_family(tmp_path, monkeypatch):
    monkeypatch.setenv(_MARKER_DIR_ENV, str(tmp_path))
    # 180 s, not 30 s. The observation window below is correctly measured from
    # the first start, but this outer ceiling is measured from launch, and one
    # child start-up costs ~16.7 s under the spawn start method on Windows. At
    # 30 s the margin was ~13 s, so a loaded machine could trip the ceiling
    # before the first start ever landed and then assert on zero starts. The
    # test still finishes in about a second past the first start in the normal
    # case; this only stops a slow start being read as a failure.
    deadline = time.monotonic() + 180
    first_seen: list[float] = []

    def done() -> bool:
        # Child start-up is slow under the spawn start method, so time the
        # observation window from the first start, not from launch.
        if not first_seen and _starts(tmp_path, WorkerFamily.ASR):
            first_seen.append(time.monotonic())
        return bool(first_seen and time.monotonic() - first_seen[0] > 1.5) or time.monotonic() > deadline

    supervise(
        [WorkerFamily.ASR], target=_exit_immediately, interval=0.05, backoff=60, should_stop=done
    )

    assert _starts(tmp_path, WorkerFamily.ASR) == 1


def test_module_constants_are_wired():
    assert task_orchestrator.WORKER_LOCK_TTL < 60 * 60
