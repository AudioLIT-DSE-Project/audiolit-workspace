"""Domain layer (SAD §5.1/§5.2) — model registry, hook manager, explanation strategies.

Skeleton stood up by LIT-227. Populated incrementally: LIT-207 (Model Registry)
moves model_loader_service.py's loading logic here first; hook_manager_service.py
(LIT-211) and the explanation-strategy services follow once the registry lands,
to avoid two in-flight branches restructuring the same files at once.
"""
