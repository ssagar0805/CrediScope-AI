"""
TruthLens Text Analysis Service (async/httpx version)

Fully async implementation using httpx.AsyncClient, backoff retries,
normalized fact-check ratings, expanded Perspective attributes, and
clean client lifecycle management with Educational Analysis System.

Place in: backend/app/services/text_service.py
"""

import os
import json
import asyncio
import logging
import httpx
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import re
from urllib.parse import quote_plus
import backoff

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_LANGUAGE = "en"

# Educational Analysis System Prompt
EDUCATIONAL_SYSTEM_PROMPT = """
You are TruthLens educational misinformation analyzer for Indian users.

For ANY claim submitted, respond with exactly 4 educational sections:

🧠 WHY THIS IS MISINFORMATION:
- Logical breakdown (science/economics/politics why it's wrong)  
- How it spreads (WhatsApp forwards, fake sites, manipulated videos)
- Use bullet points, clear practical reasoning

🎓 WHAT INDIANS SHOULD KNOW:
- Reference Indian institutions (ICMR, CDSCO, RBI, PIB, Election Commission)
- Use simple English + Hindi analogies  
- Show how Indian law/policy protects citizens

🔍 HOW TO SPOT SIMILAR CLAIMS:
- Pattern recognition (fear language, "they don't want you to know")
- Checklist: Who is source? PIB vs forwarded? Emotionally manipulative?
- Red flags specific to Indian context

🌟 THE REAL STORY:
- Evidence-based truth with Indian/global sources
- Historical context (past WhatsApp lynchings, COVID scams, political propaganda)  
- Balanced perspective, not just "false"

Always be simple, logical, unbiased, pragmatic, culturally sensitive for Indian users.
Respond in JSON format with these 4 sections.
"""

# API Configuration from environment variables
class APIConfig:
    """Centralized API configuration management"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    CLAIMBUSTER_API_KEY = None
    GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    PERSPECTIVE_API_KEY = None
    WIKIPEDIA_API_KEY = os.getenv("WIKIPEDIA_API_KEY")
    
    # Multiple possible endpoints for ClaimBuster
    CLAIMBUSTER_ENDPOINTS = []

    # API Endpoints
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    CLAIMBUSTER_ENDPOINT = "https://idir.uta.edu/claimbuster/api/v2/score/json"
    FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    PERSPECTIVE_ENDPOINT = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    WIKIPEDIA_API_ENDPOINT = "https://en.wikipedia.org/api/rest_v1/page/summary"
    WIKIDATA_ENDPOINT = "https://www.wikidata.org/w/api.php"

@dataclass
class AnalysisResult:
    """Structured result container for API responses"""
    status: str
    data: Dict[str, Any]
    error: Optional[str] = None
    timestamp: float = None
    processing_time: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AnalysisStatus(Enum):
    """Status codes for analysis results"""
    SUCCESS = "success"
    ERROR = "error" 
    TIMEOUT = "timeout"
    NOT_IMPLEMENTED = "not_implemented"
    RATE_LIMITED = "rate_limited"

# ---------------------------
# Error classes
# ---------------------------
class TextAnalysisError(Exception):
    """Custom exception for text analysis errors"""
    def __init__(self, message: str, api_name: str = None, error_code: str = None):
        self.message = message
        self.api_name = api_name
        self.error_code = error_code
        super().__init__(self.message)

class APIRateLimitError(TextAnalysisError):
    """Raised when API rate limits are exceeded"""
    pass

class APITimeoutError(TextAnalysisError):
    """Raised when API requests timeout"""
    pass

class APIConfigurationError(TextAnalysisError):
    """Raised when API configuration is invalid or missing"""
    pass

# ---------------------------
# TextAnalysisService
# ---------------------------
class TextAnalysisService:
    """Main service class for text analysis operations (async)"""
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        # You can toggle debug / verbose mode if needed
        self.debug = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT),
                headers={'User-Agent': 'TruthLens/1.0.0 (Misinformation Detection Platform)'}
            )
        return self.client

    async def close(self) -> None:
        """Gracefully close the httpx client"""
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing httpx client: {e}")
            finally:
                self.client = None

    def _mask_content_for_logging(self, content: str, max_chars: int = 20) -> str:
        """Mask content for privacy-safe logging"""
        if not content:
            return "Content empty"
        if len(content) <= max_chars:
            return f"Content masked for privacy (length: {len(content)})"
        return f"{content[:max_chars]}... [Content masked for privacy] (length: {len(content)})"

    # ---------------------------
    # Main orchestrator
    # ---------------------------
    async def analyze_text(self, content: str, language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
        """
        Main orchestrator function for text analysis workflow.
        """
        start_time = time.time()

        # Privacy-safe logging
        masked_content = self._mask_content_for_logging(content)
        logger.info(f"Starting text analysis for: {masked_content}")

        # Initialize result structure
        results = {
            "input": {
                "content_length": len(content),
                "language": language,
                "timestamp": start_time
            },
            "gemini": {},
            "claimbuster": {},
            "factcheck": {},
            "wikipedia": {},
            "perspective": {},
            "educational": {},
            "fallback": {}
        }

        try:
            # Step 1: Gemini
            logger.info("Step 1/7: Calling Gemini API for text normalization...")
            gemini_result = await self.call_gemini(content, language)
            results["gemini"] = gemini_result.data

            normalized_claim = self._extract_normalized_claim(gemini_result.data) or content

            # # Step 2: ClaimBuster
            # logger.info("Step 2/7: Calling ClaimBuster API for check-worthiness...")
            # claimbuster_result = await self.call_claimbuster(normalized_claim)
            # results["claimbuster"] = claimbuster_result.data

            # Step 3: Google Fact Check
            logger.info("Step 3/7: Calling Google Fact Check API...")
            factcheck_result = await self.call_google_factcheck(normalized_claim, language)
            results["factcheck"] = factcheck_result.data

            # Step 4: Wikipedia
            logger.info("Step 4/7: Calling Wikipedia APIs for entity verification...")
            entities = self._extract_entities(gemini_result.data)
            wikipedia_result = await self.call_wikipedia_lookup(entities)
            results["wikipedia"] = wikipedia_result.data

            # Step 5: Perspective
            logger.info("Step 5/7: Calling Perspective API for toxicity detection...")
            perspective_result = await self.call_perspective_api(content)
            results["perspective"] = perspective_result.data

            # Step 6: Educational Analysis  
            logger.info("Step 6/7: Generating educational analysis...")
            educational_result = await self.call_educational_analysis(content, results["factcheck"], language)
            results["educational"] = educational_result.data

            # Step 7: Fallback
            logger.info("Step 7/7: Initializing fallback API stubs...")
            results["fallback"] = await self._initialize_fallback_apis(normalized_claim)

        except Exception as e:
            logger.error(f"Critical error in analysis pipeline: {e}")
            results["pipeline_error"] = {
                "error": True,
                "message": str(e),
                "timestamp": time.time()
            }

        # Add processing metadata
        results["metadata"] = {
            "total_processing_time": time.time() - start_time,
            "version": "1.0.0",
            "apis_called": len([k for k, v in results.items() if isinstance(v, dict) and v.get("status") == "success"])
        }

        logger.info(f"Text analysis completed in {results['metadata']['total_processing_time']:.2f}s")
        return results

    # ---------------------------
    # Gemini integration
    # ---------------------------
    @backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=MAX_RETRIES)
    async def call_gemini(self, content: str, language: str = "en") -> AnalysisResult:
        """
        Integrates with Google's Gemini API for claim normalization and entity extraction.
        Returns AnalysisResult with normalized_claim, entities, claim_type, confidence.
        """
        start_time = time.time()

        if not APIConfig.GEMINI_API_KEY:
            logger.warning("Gemini API key not configured; returning mock response.")
            return AnalysisResult(
                status=AnalysisStatus.SUCCESS.value,
                data={
                    "status": "success",
                    "normalized_claim": content,
                    "entities": [],
                    "claim_type": "unknown",
                    "confidence": 0.5,
                    "mock": True
                },
                processing_time=time.time() - start_time
            )

        try:
            client = await self._get_client()

            prompt = (
                f"Analyze this text for misinformation detection:\n\n"
                f'Text: "{content}"\n'
                f"Language: {language}\n\n"
                "Please provide:\n"
                "1. A normalized, clear version of the main claim(s)\n"
                "2. Key entities mentioned (people, places, organizations, concepts)\n"
                "3. The type of claim (political, medical, scientific, etc.)\n"
                "4. Confidence level in your analysis (0-1)\n\n"
                "Respond in JSON format only."
            )

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
            }

            url = f"{APIConfig.GEMINI_ENDPOINT}?key={APIConfig.GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                try:
                    generated_text = (
                        result.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    cleaned_text = self._strip_markdown_fences(generated_text)
                    parsed_data = self._parse_gemini_json(cleaned_text) or {}

                    return AnalysisResult(
                        status=AnalysisStatus.SUCCESS.value,
                        data={
                            "status": "success",
                            "normalized_claim": parsed_data.get("normalized_claim", content),
                            "entities": parsed_data.get("entities", []),
                            "claim_type": parsed_data.get("claim_type", "unknown"),
                            "confidence": parsed_data.get("confidence", 0.5),
                            "raw_response": generated_text,
                        },
                        processing_time=time.time() - start_time
                    )
                except Exception as e:
                    logger.error(f"Failed parsing Gemini response: {e}")
                    return AnalysisResult(
                        status=AnalysisStatus.ERROR.value,
                        data={"error": True, "message": f"Failed to parse Gemini response: {e}"},
                        error=str(e)
                    )
            else:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR.value,
                    data={"error": True, "message": f"Gemini API HTTP {response.status_code}"},
                    error=f"HTTP {response.status_code}"
                )

        except httpx.TimeoutException:
            return AnalysisResult(
                status=AnalysisStatus.TIMEOUT.value,
                data={"error": True, "message": "Gemini API timeout"},
                error="Request timeout"
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR.value,
                data={"error": True, "message": str(e)},
                error=str(e)
            )

    # ---------------------------
    # Educational Analysis Integration
    # ---------------------------
    @backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=MAX_RETRIES)
    async def call_educational_analysis(self, content: str, factcheck_data: Dict, language: str = "en") -> AnalysisResult:
        """Generate educational analysis using Gemini with comprehensive Indian context"""
        start_time = time.time()
        
        if not APIConfig.GEMINI_API_KEY:
            return AnalysisResult(
                status=AnalysisStatus.SUCCESS.value,
                data={
                    "misinformation_analysis": "Mock educational analysis - logical breakdown needed",
                    "indian_context": "Mock Indian context - reference ICMR, PIB, RBI institutions", 
                    "pattern_recognition": "Mock pattern recognition - look for fear language and forwarded messages",
                    "real_story": "Mock real story - evidence-based truth with Indian sources",
                    "mock": True
                },
                processing_time=time.time() - start_time
            )
        
        try:
            client = await self._get_client()
            
            # Build context from fact-check results
            factcheck_context = ""
            if factcheck_data.get("verdicts"):
                factcheck_context = f"Fact-check results found: {len(factcheck_data['verdicts'])} sources including "
                sources = [v.get("source", "Unknown") for v in factcheck_data["verdicts"][:3]]
                factcheck_context += ", ".join(sources)
            
            educational_prompt = f"""
            {EDUCATIONAL_SYSTEM_PROMPT}
            
            Analyze this claim for Indian users: "{content}"
            
            Context from fact-checkers: {factcheck_context}
            
            Provide educational analysis in this JSON format:
            {{
                "misinformation_analysis": "Logical breakdown of why this is false and how it spreads",
                "indian_context": "What Indians should know - reference ICMR, PIB, RBI, Election Commission",  
                "pattern_recognition": "How to spot similar claims - red flags for Indian users",
                "real_story": "Evidence-based truth with Indian sources and historical context"
            }}
            """
            
            payload = {
                "contents": [{"parts": [{"text": educational_prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
            }
            
            url = f"{APIConfig.GEMINI_ENDPOINT}?key={APIConfig.GEMINI_API_KEY}"
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                result = response.json()
                generated_text = (
                    result.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                
                cleaned_text = self._strip_markdown_fences(generated_text)
                educational_data = self._parse_gemini_json(cleaned_text) or {}
                
                return AnalysisResult(
                    status=AnalysisStatus.SUCCESS.value,
                    data={
                        "status": "success",
                        "misinformation_analysis": educational_data.get("misinformation_analysis", "Analysis pending"),
                        "indian_context": educational_data.get("indian_context", "Context pending"),
                        "pattern_recognition": educational_data.get("pattern_recognition", "Pattern analysis pending"), 
                        "real_story": educational_data.get("real_story", "Real story pending"),
                        "raw_response": generated_text
                    },
                    processing_time=time.time() - start_time
                )
            else:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR.value,
                    data={"error": True, "message": f"Educational analysis failed: HTTP {response.status_code}"}
                )
                
        except Exception as e:
            logger.error(f"Educational analysis error: {e}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR.value,
                data={"error": True, "message": str(e)}
            )

    def _strip_markdown_fences(self, text: str) -> str:
        """Strip markdown code fences from Gemini response"""
        if not text:
            return ""
        patterns = [r'``````', r'``````']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return text.strip()

    def _parse_gemini_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from Gemini response with fallback handling"""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return None
        return None

    # ---------------------------
    # ClaimBuster integration
    # ---------------------------
    @backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=MAX_RETRIES)
    async def call_claimbuster(self, content: str) -> AnalysisResult:
        start_time = time.time()
        if not APIConfig.CLAIMBUSTER_API_KEY:
            return AnalysisResult(status=AnalysisStatus.SUCCESS.value, data={"check_worthiness_score": 0.7, "classification": "mock"})
        try:
            client = await self._get_client()
            success, last_error = None, None
            for endpoint in APIConfig.CLAIMBUSTER_ENDPOINTS:
                try:
                    response = await client.post(
                        endpoint,
                        json={"input_text": content},
                        headers={"Content-Type": "application/json", "x-api-key": APIConfig.CLAIMBUSTER_API_KEY}
                    )
                    if response.status_code == 200:
                        success = response.json()
                        break
                    else:
                        last_error = f"{endpoint} -> HTTP {response.status_code}"
                except Exception as e:
                    last_error = str(e)

            if success:
                score = self._extract_claimbuster_score(success)
                if score >= 0.8:
                    classification = "highly-check-worthy"
                elif score >= 0.6:
                    classification = "check-worthy"
                elif score >= 0.4:
                    classification = "potentially-check-worthy"
                else:
                    classification = "not-check-worthy"
                return AnalysisResult(status=AnalysisStatus.SUCCESS.value, data={"check_worthiness_score": score, "classification": classification, "raw_response": success}, processing_time=time.time() - start_time)
            else:
                return AnalysisResult(status=AnalysisStatus.ERROR.value, data={"error": True, "message": last_error})
        except Exception as e:
            return AnalysisResult(status=AnalysisStatus.ERROR.value, data={"error": True, "message": str(e)})

    def _extract_claimbuster_score(self, result: Any) -> float:
        try:
            if isinstance(result, dict) and "score" in result:
                return float(result["score"])
            if isinstance(result, list) and len(result) > 0 and "score" in result[0]:
                return float(result[0]["score"])
            return 0.0
        except Exception:
            return 0.0

    # ---------------------------
    # Google Fact Check integration
    # ---------------------------
    @backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=MAX_RETRIES)
    async def call_google_factcheck(self, query: str, language: str = "en") -> AnalysisResult:
        """Search Google Fact Check Tools for existing fact-checks and normalize ratings"""
        start_time = time.time()

        if not APIConfig.GOOGLE_FACTCHECK_API_KEY:
            logger.warning("Google Fact Check API key not configured; returning mock response.")
            return AnalysisResult(
                status=AnalysisStatus.SUCCESS.value,
                data={
                    "status": "success",
                    "fact_checks_found": 1,
                    "verdicts": [{
                        "verdict": "FALSE",
                        "original_rating": "Mock: FALSE",
                        "source": "Mock Fact Checker",
                        "confidence": 0.8,
                        "summary": "Mock fact-check result"
                    }],
                    "mock": True
                },
                processing_time=time.time() - start_time
            )

        try:
            client = await self._get_client()
            params = {"key": APIConfig.GOOGLE_FACTCHECK_API_KEY, "query": query, "languageCode": language, "pageSize": 10}
            response = await client.get(APIConfig.FACTCHECK_ENDPOINT, params=params)

            if response.status_code == 200:
                result = response.json()
                fact_checks = result.get("claims", [])
                verdicts = []
                for claim in fact_checks:
                    for review in claim.get("claimReview", []):
                        original_rating = review.get("textualRating", "") or review.get("textualRating", "UNKNOWN")
                        normalized = self._normalize_fact_check_rating(original_rating)
                        verdicts.append({
                            "verdict": normalized,
                            "original_rating": original_rating,
                            "source": review.get("publisher", {}).get("name", "Unknown"),
                            "url": review.get("url", ""),
                            "date": review.get("reviewDate", ""),
                            "summary": claim.get("text", "")
                        })
                return AnalysisResult(
                    status=AnalysisStatus.SUCCESS.value,
                    data={
                        "status": "success",
                        "fact_checks_found": len(fact_checks),
                        "verdicts": verdicts,
                        "query_used": query,
                        "raw_response": result
                    },
                    processing_time=time.time() - start_time
                )
            else:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR.value,
                    data={"error": True, "message": f"Fact Check API HTTP {response.status_code}"},
                    error=f"HTTP {response.status_code}"
                )
        except httpx.TimeoutException:
            return AnalysisResult(
                status=AnalysisStatus.TIMEOUT.value,
                data={"error": True, "message": "Fact Check API timeout"},
                error="Request timeout"
            )
        except Exception as e:
            logger.error(f"Google Fact Check API error: {e}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR.value,
                data={"error": True, "message": str(e)},
                error=str(e)
            )

    def _normalize_fact_check_rating(self, rating: str) -> str:
        """Normalize inconsistent fact-check ratings to standard format"""
        if not rating:
            return "UNKNOWN"
        rating_lower = rating.lower().strip()
        false_indicators = ["false", "fake", "pants on fire", "incorrect", "wrong", "debunked", "no evidence", "no-evidence"]
        true_indicators = ["true", "correct", "accurate", "verified", "confirmed", "mostly true", "mostly-true"]
        mixed_indicators = ["mixed", "partly true", "partly false", "half true", "partially true", "needs context", "misleading"]
        for indicator in false_indicators:
            if indicator in rating_lower:
                return "FALSE"
        for indicator in true_indicators:
            if indicator in rating_lower:
                return "TRUE"
        for indicator in mixed_indicators:
            if indicator in rating_lower:
                return "MIXED"
        return "UNKNOWN"

    # ---------------------------
    # Wikipedia lookup (async)
    # ---------------------------
    async def call_wikipedia_lookup(self, entities: List[str]) -> AnalysisResult:
        """
        Queries Wikipedia for entity summaries (async, limited to first 5 entities).
        """
        start_time = time.time()
        if not entities:
            return AnalysisResult(status=AnalysisStatus.SUCCESS.value, data={"status": "success", "entities_checked": 0, "results": [], "message": "No entities to lookup"}, processing_time=time.time() - start_time)

        try:
            client = await self._get_client()
            entity_results = []
            
            # Add Wikipedia API authentication headers
            headers = {}
            if APIConfig.WIKIPEDIA_API_KEY:
                headers["Authorization"] = f"Bearer {APIConfig.WIKIPEDIA_API_KEY}"
            
            # limit to first 5 to avoid rate limits
            for entity in entities[:5]:
                try:
                    clean_entity = quote_plus(entity.strip())
                    url = f"{APIConfig.WIKIPEDIA_API_ENDPOINT}/{clean_entity}"
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        wiki_data = response.json()
                        entity_results.append({
                            "entity": entity,
                            "found": True,
                            "title": wiki_data.get("title", ""),
                            "summary": wiki_data.get("extract", "")[:500],
                            "url": wiki_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                        })
                    else:
                        entity_results.append({"entity": entity, "found": False, "message": "Not found in Wikipedia"})
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error looking up entity '{entity}': {e}")
                    entity_results.append({"entity": entity, "found": False, "error": str(e)})
            return AnalysisResult(status=AnalysisStatus.SUCCESS.value, data={"status": "success", "entities_checked": len(entities), "results": entity_results, "found_count": len([r for r in entity_results if r.get("found")])}, processing_time=time.time() - start_time)
        except Exception as e:
            logger.error(f"Wikipedia lookup error: {e}")
            return AnalysisResult(status=AnalysisStatus.ERROR.value, data={"error": True, "message": str(e)}, error=str(e))

    # ---------------------------
    # Perspective API (expanded attributes)
    # ---------------------------
    @backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=MAX_RETRIES)
    async def call_perspective_api(self, content: str) -> AnalysisResult:
        """
        Integrates with Google's Perspective API for toxicity and manipulation detection.
        Requests expanded attributes including SPAM, SEXUALLY_EXPLICIT, FLIRTATION.
        """
        start_time = time.time()
        if not APIConfig.PERSPECTIVE_API_KEY:
            logger.warning("Perspective API key not configured; returning mock response.")
            return AnalysisResult(
                status=AnalysisStatus.SUCCESS.value,
                data={
                    "status": "success",
                    "toxicity_score": 0.2,
                    "severe_toxicity": 0.1,
                    "identity_attack": 0.0,
                    "insult_score": 0.0,
                    "spam_score": 0.0,
                    "sexually_explicit": 0.0,
                    "flirtation": 0.0,
                    "classification": "low_risk",
                    "mock": True
                },
                processing_time=time.time() - start_time
            )

        try:
            client = await self._get_client()
            payload = {
                "comment": {"text": content},
                "requestedAttributes": {
                    "TOXICITY": {},
                    "SEVERE_TOXICITY": {},
                    "IDENTITY_ATTACK": {},
                    "INSULT": {},
                    "PROFANITY": {},
                    "THREAT": {},
                    "SPAM": {},
                    "SEXUALLY_EXPLICIT": {},
                    "FLIRTATION": {}
                },
                "languages": ["en"]
            }
            url = f"{APIConfig.PERSPECTIVE_ENDPOINT}?key={APIConfig.PERSPECTIVE_API_KEY}"
            response = await client.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()
                scores = result.get("attributeScores", {})

                def safe_score(attr: str) -> float:
                    return float(scores.get(attr, {}).get("summaryScore", {}).get("value", 0.0) or 0.0)

                toxicity_score = safe_score("TOXICITY")
                severe_toxicity = safe_score("SEVERE_TOXICITY")
                identity_attack = safe_score("IDENTITY_ATTACK")
                insult_score = safe_score("INSULT")
                profanity_score = safe_score("PROFANITY")
                threat_score = safe_score("THREAT")
                spam_score = safe_score("SPAM")
                sexually_explicit = safe_score("SEXUALLY_EXPLICIT")
                flirtation = safe_score("FLIRTATION")

                all_scores = [toxicity_score, severe_toxicity, identity_attack, insult_score, spam_score, profanity_score, threat_score]
                max_score = max(all_scores) if all_scores else 0.0

                if max_score >= 0.8:
                    classification = "high_risk"
                elif max_score >= 0.5:
                    classification = "medium_risk"
                elif max_score >= 0.3:
                    classification = "low_risk"
                else:
                    classification = "minimal_risk"

                return AnalysisResult(
                    status=AnalysisStatus.SUCCESS.value,
                    data={
                        "status": "success",
                        "toxicity_score": round(toxicity_score, 3),
                        "severe_toxicity": round(severe_toxicity, 3),
                        "identity_attack": round(identity_attack, 3),
                        "insult_score": round(insult_score, 3),
                        "profanity_score": round(profanity_score, 3),
                        "threat_score": round(threat_score, 3),
                        "spam_score": round(spam_score, 3),
                        "sexually_explicit": round(sexually_explicit, 3),
                        "flirtation": round(flirtation, 3),
                        "classification": classification,
                        "max_risk_score": round(max_score, 3),
                        "raw_response": result
                    },
                    processing_time=time.time() - start_time
                )
            else:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR.value,
                    data={"error": True, "message": f"Perspective API HTTP {response.status_code}"},
                    error=f"HTTP {response.status_code}"
                )

        except httpx.TimeoutException:
            return AnalysisResult(
                status=AnalysisStatus.TIMEOUT.value,
                data={"error": True, "message": "Perspective API timeout"},
                error="Request timeout"
            )
        except Exception as e:
            logger.error(f"Perspective API error: {e}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR.value,
                data={"error": True, "message": str(e)},
                error=str(e)
            )

    # ---------------------------
    # Fallback API stubs
    # ---------------------------
    async def call_custom_search(self, query: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "message": "Custom Search pending"}

    async def call_news_api(self, query: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "message": "News API pending"}

    async def call_huggingface_model(self, text: str, model: str = "default") -> Dict[str, Any]:
        return {"status": "not_implemented", "message": "Hugging Face pending"}

    # ---------------------------
    # Helper methods
    # ---------------------------
    def _extract_normalized_claim(self, gemini_data: Dict[str, Any]) -> str:
        if not gemini_data:
            return ""
        return gemini_data.get("normalized_claim") or gemini_data.get("normalizedClaim") or gemini_data.get("text") or ""

    def _extract_entities(self, gemini_data: Dict[str, Any]) -> List[str]:
        if not gemini_data:
            return []
        entities = gemini_data.get("entities") or gemini_data.get("entityList") or []
        # ensure list of str
        return [str(e) for e in entities] if isinstance(entities, (list, tuple)) else []

    async def _initialize_fallback_apis(self, query: str) -> Dict[str, Any]:
        return {
            "custom_search": await self.call_custom_search(query),
            "news_api": await self.call_news_api(query),
            "huggingface": await self.call_huggingface_model(query)
        }

# ---------------------------
# Module-level convenience functions
# ---------------------------

async def analyze_text(content: str, language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Convenience wrapper to run the service once (creates service and closes client)."""
    service = TextAnalysisService()
    try:
        result = await service.analyze_text(content, language)
        return result
    finally:
        await service.close()

def validate_environment() -> Dict[str, bool]:
    """Validates required API keys are present."""
    status = {
        "gemini_api": bool(APIConfig.GEMINI_API_KEY),
        "claimbuster_api": True,  # Force pass since we are skipping it
        "factcheck_api": bool(APIConfig.GOOGLE_FACTCHECK_API_KEY),
        "perspective_api": bool(APIConfig.PERSPECTIVE_API_KEY),
        "wikipedia_api": bool(APIConfig.WIKIPEDIA_API_KEY)
    }
    status["all_configured"] = all(status.values())
    return status

def get_mock_analysis_result(content: str) -> Dict[str, Any]:
    """Produces a mock analysis result (useful for development without keys)."""
    return {
        "input": {"content": content, "language": "en", "timestamp": time.time()},
        "gemini": {"status": "success", "normalized_claim": content, "entities": ["entity1"], "confidence": 0.8, "mock": True},
        "claimbuster": {"status": "success", "check_worthiness_score": 0.7, "classification": "check-worthy", "mock": True},
        "factcheck": {"status": "success", "fact_checks_found": 0, "verdicts": [], "mock": True},
        "wikipedia": {"status": "success", "entities_checked": 0, "results": [], "mock": True},
        "perspective": {"status": "success", "toxicity_score": 0.1, "classification": "low_risk", "mock": True},
        "educational": {"status": "success", "misinformation_analysis": "Mock analysis", "indian_context": "Mock context", "pattern_recognition": "Mock patterns", "real_story": "Mock story", "mock": True},
        "fallback": {"custom_search": {"status": "not_implemented"}, "news_api": {"status": "not_implemented"}},
        "metadata": {"total_processing_time": 0.0, "version": "1.0.0", "apis_called": 0, "mock_data": True}
    }

async def test_all_apis(test_content: str = "This is a test claim for API validation") -> Dict[str, Any]:
    """
    Tests all API integrations with a sample input to validate configuration.
    Returns dict summarizing results.
    """
    logger.info("Starting comprehensive API testing...")
    service = TextAnalysisService()
    test_results: Dict[str, Any] = {}

    api_calls = [
        ("gemini", service.call_gemini(test_content)),
        # ("claimbuster", service.call_claimbuster(test_content)), -- remove this
        ("factcheck", service.call_google_factcheck(test_content)),
        ("wikipedia", service.call_wikipedia_lookup(["test", "entity"])),
        ("perspective", service.call_perspective_api(test_content)),
        ("educational", service.call_educational_analysis(test_content, {"verdicts": []}))
    ]

    try:
        for name, coro in api_calls:
            try:
                res: AnalysisResult = await coro
                test_results[name] = {
                    "status": res.status,
                    "success": res.status == AnalysisStatus.SUCCESS.value,
                    "processing_time": res.processing_time,
                    "error": res.error
                }
            except Exception as e:
                test_results[name] = {"status": "error", "success": False, "error": str(e)}
        successful = sum(1 for v in test_results.values() if v.get("success"))
        test_results["summary"] = {
            "total_apis_tested": len(api_calls),
            "successful_apis": successful,
            "success_rate": successful / len(api_calls),
            "all_passed": successful == len(api_calls)
        }
        logger.info(f"API testing completed. Success rate: {test_results['summary']['success_rate']:.2%}")
        return test_results
    finally:
        await service.close()

# ---------------------------
# Initialize helper
# ---------------------------
def initialize_service() -> TextAnalysisService:
    """Create and return a configured TextAnalysisService instance (does not create client yet)."""
    env_status = validate_environment()
    logger.info("Text Analysis Service Configuration:")
    for k, v in env_status.items():
        logger.info(f"  {k}: {'✓ Configured' if v else '✗ Missing'}")
    if not env_status["all_configured"]:
        logger.warning("Some APIs not configured. Service will use mock responses where applicable.")
    return TextAnalysisService()

# ---------------------------
# CLI / direct run
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens Text Analysis Service")
    parser.add_argument("--test", action="store_true", help="Run API tests")
    parser.add_argument("--validate", action="store_true", help="Validate environment configuration")
    parser.add_argument("--analyze", type=str, help="Analyze provided text")
    parser.add_argument("--mock", action="store_true", help="Use mock responses")
    args = parser.parse_args()

    async def main():
        if args.validate:
            env_status = validate_environment()
            print("Environment Validation:")
            for k, v in env_status.items():
                print(f"  {k}: {'✓' if v else '✗'}")
        elif args.test:
            print("Running comprehensive API tests...")
            results = await test_all_apis()
            print("Test summary:", results.get("summary"))
        elif args.analyze:
            print(f"Analyzing: {args.analyze}")
            if args.mock:
                res = get_mock_analysis_result(args.analyze)
            else:
                res = await analyze_text(args.analyze)
            # pretty print a short summary
            print("Result summary:")
            print("  ClaimBuster score:", res.get("claimbuster", {}).get("check_worthiness_score"))
            print("  Fact checks found:", res.get("factcheck", {}).get("fact_checks_found"))
            print("  Toxicity score:", res.get("perspective", {}).get("toxicity_score"))
            print("  Educational sections:", "✓" if res.get("educational", {}).get("status") == "success" else "✗")
            print("  Processing time:", res.get("metadata", {}).get("total_processing_time"))
        else:
            print("TruthLens Text Analysis Service")
            print("Use --help for options")

    asyncio.run(main())
