"""
app.py  —  Startup shim
========================
This file exists solely so that IIS / web.config can keep pointing to
``app.py`` as the entry point.

It loads the .env file, then hands off to uvicorn which serves the FastAPI
application defined in ``app/main.py``.

Previously this file contained the full Flask application; all business logic
has been moved to the ``app/`` package.
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Re-export the FastAPI app so that tools referencing ``app.app`` work too.
from app.main import app  # noqa: F401, E402

if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PLATFORM_PORT", 5002))
    print(f"Starting Face Recognition API (FastAPI) on port {port}...")
    print(f"  Dataiku Model API URL: {os.environ.get('DATAIKU_MODEL_API_URL', 'http://localhost:5003')}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # keep single-worker for IIS HttpPlatformHandler compatibility
    )
