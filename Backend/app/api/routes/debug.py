from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.infrastructure.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/debug/session")
async def get_session_info(request: Request):
    """Debug endpoint to see the current session id (LIT-223).

    This endpoint used to echo back the request's cookies and all headers to
    any caller, which SAD §11.3 flagged as an inherited weakness ("a diagnostic
    feature that exposed too much information"). It is now gated behind
    ``DEBUG_ENABLED`` (default off) and returns only the session id when
    enabled - never cookies or headers.
    """
    if not settings.DEBUG_ENABLED:
        raise HTTPException(status_code=404, detail="Debug endpoints are disabled")
    session_id = getattr(request.state, 'sid', None)

    return JSONResponse({"session_id": session_id})