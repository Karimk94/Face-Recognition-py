"""
routers/analyze.py
==================
Image analysis endpoints:

  POST /api/analyze_image         — Multipart form-file upload
  POST /api/analyze_image_stream  — Raw byte stream in the request body
"""

import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.services.image import process_image_via_dataiku

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze_image")
async def api_analyze_image(image_file: UploadFile = File(...)):
    """Process an image uploaded as a multipart form file."""
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image file provided.")

    try:
        results = process_image_via_dataiku(image_bytes)
        return JSONResponse(content=results)
    except Exception as exc:
        logger.error("Error in /analyze_image: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc


@router.post("/analyze_image_stream")
async def api_analyze_image_stream(request: Request):
    """Process a single image sent as a raw byte stream."""
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data in request body.")

    try:
        results = process_image_via_dataiku(image_bytes)
        return JSONResponse(content=results)
    except Exception as exc:
        logger.error("Error in /analyze_image_stream: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc
