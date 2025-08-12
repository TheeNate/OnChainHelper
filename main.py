from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from collections import defaultdict

# Request models
class MetricsRequest(BaseModel):
    endpoint_path: str = Field(..., description="Path under https://api.researchbitcoin.net")
    query: Optional[Dict[str, Any]] = Field(default=None, description="Optional query parameters")

# Rate limiting storage (in-memory for simplicity)
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # 1 minute

# Configuration
RESEARCHBITCOIN_BASE_URL = "https://api.researchbitcoin.net"
RESEARCHBITCOIN_TOKEN = os.environ.get("RESEARCHBITCOIN_TOKEN")

if not RESEARCHBITCOIN_TOKEN:
    print("WARNING: RESEARCHBITCOIN_TOKEN environment variable is not set")

# Create FastAPI instance
app = FastAPI(title="Financial Data API Proxy", version="1.0.0")

# Add CORS middleware - restrict to ChatGPT and test origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat.openai.com",
        "https://chatgpt.com", 
        "http://localhost:*",
        "https://*.replit.app",
        "https://*.replit.dev"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxy headers"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(client_ip: str) -> bool:
    """Simple in-memory rate limiting: 30 requests per minute per IP"""
    now = time.time()
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if under limit
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

def validate_endpoint_path(path: str) -> bool:
    """Validate endpoint path for security"""
    # Must start with /
    if not path.startswith("/"):
        return False
    
    # Length limit
    if len(path) > 200:
        return False
    
    # No path traversal
    if ".." in path:
        return False
    
    # Only allow safe characters: alphanumeric, /, _, -
    if not re.match(r'^[a-zA-Z0-9/_-]+$', path):
        return False
    
    # Must not be a full URL
    if path.startswith("http://") or path.startswith("https://"):
        return False
    
    return True

def redact_token_from_url(url: str) -> str:
    """Redact token from URL for logging/response"""
    return re.sub(r'token=[^&]*', 'token=***REDACTED***', url)

@app.get("/healthz")
async def health_check() -> JSONResponse:
    """
    Health check endpoint that returns a simple OK status.
    """
    return JSONResponse(
        content={"ok": True},
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

@app.get("/status")
async def status_check() -> JSONResponse:
    """
    Alternative status check endpoint that returns system status.
    Added as backup for environments that might interfere with /healthz.
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Financial Data API",
            "version": "1.0.0",
            "ok": True
        },
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

@app.post("/tool/get_metrics")
async def get_metrics(request: Request, metrics_request: MetricsRequest) -> JSONResponse:
    """
    Secure proxy endpoint for ResearchBitcoin API that fetches real metrics.
    """
    # Rate limiting
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail={"ok": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded: 30 requests per minute"}}
        )
    
    # Validate endpoint path
    if not validate_endpoint_path(metrics_request.endpoint_path):
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": {"code": "INVALID_PATH", "message": "Invalid endpoint_path: must start with /, no path traversal, max 200 chars, alphanumeric/underscore/dash only"}}
        )
    
    # Check if token is available
    if not RESEARCHBITCOIN_TOKEN:
        raise HTTPException(
            status_code=502,
            detail={"ok": False, "error": {"code": "UPSTREAM_ERROR", "message": "ResearchBitcoin token not configured"}}
        )
    
    # Build query parameters
    query_params = metrics_request.query or {}
    
    # Default output_format to json if not provided
    if "output_format" not in query_params:
        query_params["output_format"] = "json"
    
    # Default date_field to yesterday's date in UTC if missing or empty
    # (ResearchBitcoin API requires dates up to yesterday only)
    date_field = query_params.get("date_field")
    if not date_field or (isinstance(date_field, str) and date_field.strip() == ""):
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        query_params["date_field"] = yesterday.strftime("%Y-%m-%d")
    
    # Always add the token (from environment, not client)
    query_params["token"] = RESEARCHBITCOIN_TOKEN
    
    # Build upstream URL
    upstream_url = f"{RESEARCHBITCOIN_BASE_URL}{metrics_request.endpoint_path}"
    
    try:
        # Make request to upstream API with 10s timeout
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                upstream_url,
                params=query_params,
                headers={"Accept": "application/json"}
            )
            
            # Handle upstream errors
            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "ok": False, 
                        "error": {
                            "code": "UPSTREAM_ERROR", 
                            "message": f"Upstream API returned {response.status_code}: {response.text[:200]}"
                        }
                    }
                )
            
            # Parse response JSON
            try:
                upstream_data = response.json()
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "ok": False,
                        "error": {
                            "code": "UPSTREAM_ERROR",
                            "message": f"Failed to parse upstream JSON: {str(e)}"
                        }
                    }
                )
            
            # Return successful response with metadata
            return JSONResponse(
                content={
                    "ok": True,
                    "source": redact_token_from_url(str(response.url)),
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "data": upstream_data
                },
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": "Upstream API request timed out after 10 seconds"
                }
            }
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": f"Failed to connect to upstream API: {str(e)}"
                }
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": f"Unexpected error: {str(e)}"
                }
            }
        )

# Root endpoint for basic API info
@app.get("/")
async def root() -> JSONResponse:
    """
    Root endpoint providing basic API information.
    """
    return JSONResponse(
        content={
            "message": "Financial Data API Proxy",
            "version": "1.0.0",
            "description": "Secure proxy for ResearchBitcoin API",
            "endpoints": {
                "health": "/healthz",
                "status": "/status", 
                "metrics": "/tool/get_metrics"
            },
            "rate_limit": "30 requests per minute per IP"
        },
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

if __name__ == "__main__":
    import os
    
    # Get port from environment variable or default to 5000
    # This allows flexibility in deployment environments
    port = int(os.environ.get("PORT", 5000))
    
    # Run the application
    # Bind to 0.0.0.0 to make it accessible externally
    # Removed reload=True for production deployment compatibility
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
