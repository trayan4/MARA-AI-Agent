"""
FastAPI application for MARA.
Provides REST API endpoints for querying the MARA system.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import uvicorn
from pathlib import Path
import shutil
from datetime import datetime

from loguru import logger

from config import settings
from orchestrator.graph import execute_query, get_mara_graph
from agents.rag import get_rag_agent
from agents.data import get_data_agent
from tools.local_vector_store import get_vector_store
from tools.chunking import chunk_text


# Pydantic Models for API

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="User query to process", min_length=1)
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")
    include_metadata: bool = Field(default=True, description="Include execution metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze the sales trends in the uploaded data",
                "context": {"uploaded_files": ["data/sales_q4.csv"]},
                "include_metadata": True
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    query: str
    executive_summary: str
    visual_insights: Dict[str, Any]
    data_insights: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class AddDocumentRequest(BaseModel):
    """Request model for adding documents to knowledge base."""
    
    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "company_policy_2024",
                "content": "Our company policy states...",
                "metadata": {"author": "HR", "date": "2024-01-01", "category": "policy"}
            }
        }


class SearchRequest(BaseModel):
    """Request model for document search."""
    
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results", ge=1, le=20)
    use_hybrid: bool = Field(default=True, description="Use hybrid search")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    version: str
    components: Dict[str, str]
    timestamp: str


# Initialize FastAPI app

app = FastAPI(
    title="MARA API",
    description="Multimodal Agentic Reasoning Assistant - AI-powered document, image, and data analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
if settings.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Startup/Shutdown Events

# @app.on_event("startup")
# async def startup_event():
#     """Initialize services on startup."""
#     logger.info("Starting MARA API server...")
    
#     # Ensure data directories exist
#     for path_attr in ["data_dir", "uploads_dir", "logs_dir"]:
#         path = Path(getattr(settings.paths, path_attr))
#         path.mkdir(parents=True, exist_ok=True)
    
#     # Initialize graph (lazy loading)
#     # get_mara_graph()
    
#     logger.info("MARA API server started successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting MARA API server...")
    
    # Ensure data directories exist
    for path_attr in ["data_dir", "uploads_dir", "logs_dir"]:
        path = Path(getattr(settings.paths, path_attr))
        path.mkdir(parents=True, exist_ok=True)
    
    # Configure file logging
    log_path = Path(settings.paths.logs_dir) / "mara.log"
    logger.add(
        log_path,
        rotation="100 MB",      # Rotate when file reaches 100MB
        retention="30 days",    # Keep logs for 30 days
        compression="zip",      # Compress old logs
        level="INFO",           # Log level
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info("File logging configured: {}", log_path)
    logger.info("MARA API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MARA API server...")


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to MARA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and component availability.
    """
    components = {
        "orchestrator": "ok",
        "vector_store": "ok",
        "llm": "ok"
    }
    
    # Test components
    try:
        get_vector_store()
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"
    
    try:
        get_mara_graph()
    except Exception as e:
        components["orchestrator"] = f"error: {str(e)}"
    
    overall_status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a query through MARA system.
    
    This is the main endpoint that orchestrates all agents to answer queries.
    """
    try:
        logger.info(f"Processing query: '{request.query[:100]}...'")
        
        # Execute query through orchestrator
        result = execute_query(
            query=request.query,
            context=request.context
        )
        
        # Check if execution was successful
        if not result.get('executive_summary'):
            raise HTTPException(
                status_code=500,
                detail=f"Query execution failed: {result.get('error', 'Unknown error')}"
            )
        
        # Build response
        response = QueryResponse(
            query=result.get('query', request.query),
            executive_summary=result.get('executive_summary', ''),
            visual_insights=result.get('visual_insights', {}),
            data_insights=result.get('data_insights', {}),
            evidence=result.get('evidence', []),
            recommendations=result.get('recommendations', []),
            confidence=result.get('confidence', 0.0),
            metadata=result.get('metadata') if request.include_metadata else None
        )
        
        logger.info(f"Query processed successfully (confidence: {response.confidence:.2f})")
        return response
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (image, document, data) for analysis.
    Returns the file path that can be used in queries.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save file
        upload_dir = Path(settings.paths.uploads_dir)
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        return {
            "filename": file.filename,
            "filepath": str(file_path),
            "size": file_path.stat().st_size,
            "message": "File uploaded successfully. Use 'filepath' in your query context."
        }
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/add", tags=["Knowledge Base"])
async def add_document(request: AddDocumentRequest):
    """
    Add a document to the knowledge base.
    The document will be chunked and embedded for retrieval.
    """
    try:
        rag_agent = get_rag_agent()
        
        success = rag_agent.add_document(
            doc_id=request.doc_id,
            content=request.content,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document")
        
        return {
            "doc_id": request.doc_id,
            "status": "added",
            "message": f"Document '{request.doc_id}' added to knowledge base"
        }
    
    except Exception as e:
        logger.error(f"Document addition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", tags=["Knowledge Base"])
async def delete_document(doc_id: str):
    """
    Delete a document from the knowledge base.
    """
    try:
        rag_agent = get_rag_agent()
        
        success = rag_agent.delete_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        return {
            "doc_id": doc_id,
            "status": "deleted",
            "message": f"Document '{doc_id}' removed from knowledge base"
        }
    
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Knowledge Base"])
async def list_documents(limit: int = 10):
    """
    List documents in the knowledge base.
    """
    try:
        rag_agent = get_rag_agent()
        documents = rag_agent.list_documents(limit=limit)
        
        return {
            "count": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", tags=["Knowledge Base"])
async def search_documents(request: SearchRequest):
    """
    Search documents in the knowledge base.
    Returns relevant chunks without full answer generation.
    """
    try:
        rag_agent = get_rag_agent()
        
        results = rag_agent.search_documents(
            query=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def get_stats():
    """
    Get system statistics.
    """
    try:
        vector_store = get_vector_store()
        rag_agent = get_rag_agent()
        
        stats = {
            "vector_store": vector_store.get_stats(),
            "rag_agent": rag_agent.get_stats()
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/data", tags=["Analysis"])
async def analyze_data(
    file: UploadFile = File(...),
    analysis_type: str = Form("describe")
):
    """
    Quick data analysis endpoint.
    Upload a CSV/Excel file and get immediate analysis.
    """
    try:
        # Save uploaded file
        upload_dir = Path(settings.paths.uploads_dir)
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze
        data_agent = get_data_agent()
        result = data_agent.analyze_dataframe(
            data_path=file_path,
            analysis_type=analysis_type
        )
        
        return result.to_dict()
    
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools", tags=["System"])
async def list_tools():
    """
    List available tools/agents and their capabilities.
    """
    from agents.planner import get_planner
    
    try:
        planner = get_planner()
        
        return {
            "agents": {
                "rag": "Document retrieval and question answering",
                "vision": "Image analysis, chart extraction, OCR",
                "data": "Statistical analysis, trend detection, anomaly detection",
                "web_search": "Real-time web search for current information"
            },
            "tools": planner.tools
        }
    
    except Exception as e:
        logger.error(f"Tool listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Run the application

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=1 if settings.api.reload else settings.api.workers
    )


if __name__ == "__main__":
    run_server()