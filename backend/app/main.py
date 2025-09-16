from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.models import HealthResponse
from app.verify import router as verify_router
from app.config import settings
from app.routes import text_analysis
# Initialize FastAPI app
app = FastAPI(
    title="TruthLens API",
    description="AI-powered misinformation detection platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React dev server (Create React App)
        "http://localhost:5173",     # Vite dev server  
        settings.FRONTEND_ORIGIN,    # Configurable origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TruthLens API - Misinformation Detection Platform",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "verify": "/api/v1/verify",
            "results": "/api/v1/results/{analysis_id}",
            "archive": "/api/v1/archive"
        }
    }

# Include API routes with versioning
app.include_router(verify_router, prefix="/api/v1", tags=["verification"])
app.include_router(text_analysis.router, prefix="/api/v1", tags=["text"])

# Additional health check for specific path
@app.get("/api/v1/health")
async def api_health():
    """API-specific health check"""
    return {"status": "healthy", "api_version": "v1", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    )