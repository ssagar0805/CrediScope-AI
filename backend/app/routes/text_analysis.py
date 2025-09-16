"""
FastAPI Router for Text Analysis
Exposes POST /text/analyze endpoint that calls AnalysisEngine
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

from app.services.analysis_engine import analysis_engine

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/text",
    tags=["Text"]
)

# Request Models
class TextAnalyzeRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text content to analyze")
    language: Optional[str] = Field("en", description="Language code (e.g., 'en', 'hi')")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Drinking bleach cures COVID-19",
                "language": "en"
            }
        }

# Response Models
class LegacyAnalysisResponse(BaseModel):
    """Response model matching AnalysisEngine output"""
    verdict: str = Field(..., description="Analysis verdict: true, false, or inconclusive")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    summary: str = Field(..., description="Summary of analysis results")
    processing_time: float = Field(..., description="Processing time in seconds")
    detailed_analysis: Dict[str, Any] = Field(..., description="Detailed analysis from all services")
    
    class Config:
        schema_extra = {
            "example": {
                "verdict": "false",
                "confidence_score": 0.95,
                "summary": "Fact-checks found: Snopes -> FALSE",
                "processing_time": 2.34,
                "detailed_analysis": {
                    "factcheck": {
                        "fact_checks_found": 1,
                        "verdicts": [{"verdict": "FALSE", "source": "Snopes"}]
                    },
                    "gemini": {"normalized_claim": "Drinking bleach cures COVID-19"},
                    "perspective": {"toxicity_score": 0.1}
                }
            }
        }

class PingResponse(BaseModel):
    """Simple ping response"""
    status: str
    message: str

# Routes
@router.get("/ping", response_model=PingResponse)
async def ping():
    """Health check for text analysis service"""
    return PingResponse(status="healthy", message="Text analysis service is operational")

@router.post("/analyze", response_model=LegacyAnalysisResponse)
async def analyze_text(request: TextAnalyzeRequest):
    """
    Analyze text for misinformation detection
    
    - **text**: The text content to analyze (required)
    - **language**: Language code for the text (optional, defaults to 'en')
    
    Returns verdict, confidence score, summary, and detailed analysis results.
    """
    try:
        logger.info(f"Analyzing text (length: {len(request.text)}, lang: {request.language})")
        
        # Call the analysis engine
        result = await analysis_engine.process_content(
            content_type="text",
            content=request.text,
            language=request.language or "en"
        )
        
        logger.info(f"Analysis completed: verdict={result['verdict']}, confidence={result['confidence_score']}")
        
        # Return the result (already matches LegacyAnalysisResponse structure)
        return LegacyAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis failed",
                "message": str(e),
                "type": "internal_error"
            }
        )