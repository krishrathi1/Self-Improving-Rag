from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from services.ingestion_service import get_ingestion_service

router = APIRouter()

from app.models import IngestRequest

class IngestResponse(BaseModel):
    status: str
    message: str
    doc_id: Optional[str] = None
    stats: Optional[dict] = None

@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest):
    """
    Ingest text directly into the system.
    """
    service = get_ingestion_service()
    try:
        stats = await service.ingest_text(request.content, request.source)
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {request.source}",
            doc_id=stats["doc_id"],
            stats=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file (txt/md) into the system.
    """
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported for now.")
    
    content = await file.read()
    text = content.decode("utf-8")
    
    service = get_ingestion_service()
    try:
        stats = await service.ingest_text(text, file.filename)
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
