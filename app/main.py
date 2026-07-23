"""
app/main.py
===========
FastAPI application factory.

Creates the ``FastAPI`` instance, attaches CORS middleware, and registers
all routers.  Import the ``app`` object from here when running with uvicorn:

    uvicorn app.main:app --host 0.0.0.0 --port 5002
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, faces

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Thumbnail API (Face Recognition Gateway)",
    description=(
        "A lightweight API that serves as the frontend gateway for face analysis. "
        "Handles all image manipulation (cropping, thumbnails, bounding boxes) and "
        "delegates ML inference to the Dataiku Model API."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(analyze.router)
app.include_router(faces.router)
