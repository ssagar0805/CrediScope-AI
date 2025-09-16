from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import base64
import uuid
import json
import asyncio

from pydantic import BaseModel
from app.models import AnalysisResponse
from app.services.text_service import analyze_text
from app.database import storage

router = APIRouter()

class VerifyRequest(BaseModel):
    content_type: str                
    content: str                     
    language: Optional[str] = "en"   
    user_id: Optional[str] = None    

@router.post("/verify", response_model=AnalysisResponse)
async def verify_content(request: VerifyRequest):
    """Core content verification endpoint"""
    content_type = request.content_type
    content = request.content
    language = request.language
    user_id = request.user_id

    # Validate content_type
    if content_type not in {"text", "url", "image"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid content_type. Must be 'text', 'url', or 'image'"
        )

    try:
        # If image, decode base64 to validate
        if content_type == "image":
            try:
                image_bytes = base64.b64decode(content)
                content = base64.b64encode(image_bytes).decode('utf-8')
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 image data"
                )

        # Call educational text analysis
        analysis_result = await analyze_text(content, language)
        
        # 🔍 DEBUG PRINTS - TO FIND THE ISSUE
        print("🔍 DEBUG - Full analysis_result keys:", list(analysis_result.keys()))
        print("🔍 DEBUG - Educational data:", analysis_result.get("educational"))
        print("🔍 DEBUG - Educational data type:", type(analysis_result.get("educational")))
        print("🔍 DEBUG - Educational data content:", analysis_result.get("educational", {}))
        
        # Map text_service.py response to expected format
        formatted_result = {
            "verdict": "inconclusive",  # Default value
            "confidence_score": analysis_result.get("gemini", {}).get("confidence", 0.5),
            "summary": "Analysis completed",  # Default value
            "processing_time": analysis_result.get("metadata", {}).get("total_processing_time", 0),
            "detailed_analysis": {
                "evidence": ["Analysis completed"],
                "sources": ["Google Gemini AI", "Google Perspective API"], 
                "gemini_analysis": str(analysis_result.get("gemini", {})),
                "factcheck_results": analysis_result.get("factcheck", {}).get("verdicts", []),
                "vision_analysis": None,
                "educational": analysis_result.get("educational", {})  # 🎯 ADD EDUCATIONAL DATA
            }
        }
        
        # 🔍 DEBUG PRINT - Final educational data being sent
        print("🔍 DEBUG - Final educational in response:", formatted_result["detailed_analysis"]["educational"])

        # Generate ID and build response
        analysis_id = str(uuid.uuid4())
        response = AnalysisResponse(
            analysis_id=analysis_id,
            verdict=formatted_result["verdict"],
            confidence_score=formatted_result["confidence_score"],
            summary=formatted_result["summary"],
            processing_time=formatted_result["processing_time"],
            detailed_analysis=formatted_result["detailed_analysis"]
        )

        # Persist to storage
        storage_data = {
            "analysis_id": analysis_id,
            "content_type": content_type,
            "content": "[IMAGE_DATA]" if content_type == "image" else content,
            "language": language,
            "user_id": user_id,
            "verdict": response.verdict,
            "confidence_score": response.confidence_score,
            "summary": response.summary,
            "processing_time": response.processing_time,
            "detailed_analysis": response.detailed_analysis.dict()
        }
        saved = storage.save_analysis(analysis_id, storage_data)
        if not saved:
            print(f"Warning: Failed to save analysis {analysis_id}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"🚨 DEBUG - Exception in verify_content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# 🚀 NEW: Streaming endpoint for real-time analysis
@router.post("/verify-stream")
async def verify_content_stream(request: Request):
    """Real-time streaming analysis endpoint like ChatGPT"""
    try:
        body = await request.json()
        content = body.get("content", "")
        language = body.get("language", "en")
        
        async def generate_analysis_stream():
            try:
                # Step 1: Start
                yield f"data: {json.dumps({'type': 'message', 'content': '🚀 Starting analysis...'})}\n\n"
                await asyncio.sleep(1)
                
                # Step 2: Get full analysis (reuse existing function)
                yield f"data: {json.dumps({'type': 'message', 'content': '🧠 Analyzing with AI and gathering educational insights...'})}\n\n"
                analysis_result = await analyze_text(content, language)
                await asyncio.sleep(1)
                
                # Step 3: Stream educational sections one by one
                educational_data = analysis_result.get("educational", {})
                sections = [
                    ("misinformation_analysis", "🧠 Why This is Misinformation"),
                    ("indian_context", "🎓 What Indians Should Know"), 
                    ("pattern_recognition", "🔍 How to Spot Similar Claims"),
                    ("real_story", "🌟 The Real Story")
                ]
                
                for section_key, section_title in sections:
                    yield f"data: {json.dumps({'type': 'message', 'content': f'📝 Generating {section_title}...'})}\n\n"
                    await asyncio.sleep(0.5)
                    
                    section_data = educational_data.get(section_key, {})
                    yield f"data: {json.dumps({'type': 'section', 'section': section_key, 'title': section_title, 'data': section_data})}\n\n"
                    await asyncio.sleep(2)  # 2-second smooth delay as requested
                
                # Step 4: Complete
                yield f"data: {json.dumps({'type': 'complete', 'content': '✅ Analysis complete!'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
        
        return StreamingResponse(
            generate_analysis_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@router.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Retrieve detailed analysis results by ID"""
    try:
        result = storage.get_analysis(analysis_id)
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return result
    except Exception as e:
        if "Analysis not found" in str(e):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.get("/archive")
async def get_archive(limit: int = 20, user_id: Optional[str] = None):
    """Get analysis archive/history"""
    try:
        analyses = storage.get_all_analyses(limit)
        if user_id:
            analyses = [a for a in analyses if a.get("user_id") == user_id]
        return {
            "analyses": analyses,
            "total": len(analyses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve archive: {str(e)}")
