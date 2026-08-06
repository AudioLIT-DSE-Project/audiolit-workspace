"""Orchestration layer (SAD §5.1 / §5.2 / §6.1) — RQ-based task fan-out/fan-in.

`task_orchestrator.py` is the SAD §5.2 Task Orchestrator: the single queue
fabric, worker and enqueue API. LIT-230 consolidated it from two parallel
implementations (`rq_broker.py` and `services/queue_service.py`) that had both
landed on `develop`; do not add a third — extend this one.

Also here:
  * `multitask_orchestrator_service.py` — the real ASR+SER+ADD fan-out (LIT-150).
  * `fanout_orchestrator_service.py` — LIT-225's fan-out/fan-in reference pattern.
  * `session_queue_service.py` — the per-session list queue behind `/queue`
    (unrelated to the RQ fabric, despite the name it used to share with it).
"""
