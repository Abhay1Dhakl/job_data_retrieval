from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    """Return a simple health check payload.

    Returns:
        A JSON-serializable status dict.
    """
    return {"status": "ok"}
