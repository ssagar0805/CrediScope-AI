from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class AnalysisRequest(BaseModel):
    content_type: str = Field(..., description="Type of content: text, url, or image")
    content: str = Field(..., description="Content to analyze")
    language: Optional[str] = Field("en", description="Language preference: hi or en")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class DetailedAnalysis(BaseModel):
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the verdict")
    sources: List[str] = Field(default_factory=list, description="Sources checked during analysis")
    gemini_analysis: Optional[str] = Field(None, description="AI-generated explanation")
    factcheck_results: List[Dict[str, Any]] = Field(default_factory=list, description="Fact check results")
    vision_analysis: Optional[str] = Field(None, description="Image analysis results")
    educational: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Educational analysis with misinformation breakdown")  # 🎯 ADDED EDUCATIONAL FIELD

class AnalysisResponse(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis identifier")
    verdict: str = Field(..., description="Analysis verdict: true, false, or inconclusive")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    summary: str = Field(..., description="Brief analysis summary")
    processing_time: float = Field(..., description="Time taken for analysis in seconds")
    detailed_analysis: DetailedAnalysis = Field(default_factory=DetailedAnalysis)

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
