from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import Any, Dict

# Create FastAPI instance
app = FastAPI(title="Financial Data API", version="1.0.0")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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

@app.post("/tool/get_confluence")
async def get_confluence(request_body: Dict[str, Any] = {}) -> JSONResponse:
    """
    Mock confluence endpoint that returns financial data.
    Ignores the request body and returns mock financial data as specified.
    """
    # Mock financial data as requested
    mock_data = {
        "ok": True,
        "data": {
            "score": 4,
            "components": {
                "mvrv_z": {
                    "value": 1.2,
                    "state": "neutral"
                },
                "sopr_all_ema7": {
                    "value": 1.01,
                    "state": "bullish"
                },
                "sth_mvrv": {
                    "value": 1.05,
                    "state": "neutral"
                },
                "lth_cdd": {
                    "value": 0.8,
                    "state": "neutral"
                },
                "sellside_risk": {
                    "value": 0.35,
                    "state": "caution"
                },
                "funding": {
                    "value": 0.01,
                    "state": "neutral"
                },
                "oi": {
                    "value": 72.0,
                    "state": "neutral"
                }
            },
            "rationale": "Mock confluence for wiring tests.",
            "ts": "2025-08-12T16:00:00Z"
        }
    }
    
    return JSONResponse(
        content=mock_data,
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

# Root endpoint for basic API info
@app.get("/")
async def root() -> JSONResponse:
    """
    Root endpoint providing basic API information.
    """
    return JSONResponse(
        content={
            "message": "Financial Data API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/healthz",
                "confluence": "/tool/get_confluence"
            }
        },
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

if __name__ == "__main__":
    # Run the application on port 5000 for Replit compatibility
    # Bind to 0.0.0.0 to make it accessible externally
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
