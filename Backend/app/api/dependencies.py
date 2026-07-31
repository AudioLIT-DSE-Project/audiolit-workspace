from typing import Optional

from fastapi import HTTPException, Request


def get_session_id(request: Request) -> Optional[str]:
    """Extract the session ID set by infrastructure.session.SessionMiddleware."""
    return getattr(request.state, "sid", None)


def require_session_id(request: Request) -> str:
    """Like get_session_id, but raises 400 if no session was established."""
    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID found")
    return session_id
