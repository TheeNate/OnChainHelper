from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx
import os
import re
import time
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List
from collections import defaultdict

# Request models
class MetricsRequest(BaseModel):
    endpoint_path: str = Field(..., description="Path under https://api.researchbitcoin.net")
    query: Optional[Dict[str, Any]] = Field(default=None, description="Optional query parameters")

class NamedMetricsRequest(BaseModel):
    indicators: List[str] = Field(..., description="List of indicator names (friendly names or slugs)")
    date_field: Optional[str] = Field(default=None, description="Date in YYYY-MM-DD format (defaults to yesterday)")
    window: Optional[str] = Field(default=None, description="Time window (e.g., '1d', '7d', '30d')")

# Indicator index for friendly name resolution
class IndicatorEntry(BaseModel):
    endpoint_path: str
    data_field: str
    category: str
    description: str = ""

# Rate limiting storage (in-memory for simplicity)
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # 1 minute

# Configuration
RESEARCHBITCOIN_BASE_URL = "https://api.researchbitcoin.net"
RESEARCHBITCOIN_TOKEN = os.environ.get("RESEARCHBITCOIN_TOKEN")

if not RESEARCHBITCOIN_TOKEN:
    print("WARNING: RESEARCHBITCOIN_TOKEN environment variable is not set")

# Indicator index storage
INDICATOR_INDEX: Dict[str, IndicatorEntry] = {}
SWAGGER_CACHE = None
SWAGGER_LAST_FETCH = None
SWAGGER_CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds

# Static aliases for common indicators
INDICATOR_ALIASES = {
    # MVRV aliases
    "mvrv": "mvrv",
    "mvrv_ratio": "mvrv", 
    "market_value_realized_value": "mvrv",
    
    # SOPR aliases  
    "sopr": "sopr",
    "spent_output_profit_ratio": "sopr",
    "sth_sopr": "sopr_sth",
    "short_term_holder_sopr": "sopr_sth",
    "lth_sopr": "sopr_lth", 
    "long_term_holder_sopr": "sopr_lth",
    
    # Supply aliases
    "supply": "supply_circulating",
    "total_supply": "supply_circulating",
    "circulating_supply": "supply_circulating",
    "coinbase_supply": "supply_coinbase",
    
    # Price aliases
    "price": "price_usd",
    "btc_price": "price_usd",
    "bitcoin_price": "price_usd",
    
    # Hash rate aliases
    "hashrate": "hash_rate",
    "hash_rate_7d": "hash_rate_7d_ma",
    "mining_difficulty": "difficulty",
    
    # Network aliases
    "active_addresses": "addresses_active",
    "new_addresses": "addresses_new",
    "transaction_count": "transactions_count",
    "tx_count": "transactions_count",
    
    # Additional common aliases
    "aviv": "aviv-nupl",  # AVIV NUPL indicator
    "nupl": "aviv-nupl",
    "aviv_nupl": "aviv-nupl",
    "nrpl": "realized_profit_loss_ratio",
    "profit_loss": "coindays_in_profit",
    "loss_profit": "coindays_in_loss",
    
    # Common abbreviations
    "btc": "supply_circulating",
    "market_cap": "marketcap",
    "mcap": "marketcap",
}

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

async def fetch_swagger_spec() -> Optional[Dict]:
    """Fetch swagger specification from ResearchBitcoin API"""
    global SWAGGER_CACHE, SWAGGER_LAST_FETCH
    
    try:
        # Check if cache is still valid
        now = time.time()
        if (SWAGGER_CACHE is not None and 
            SWAGGER_LAST_FETCH is not None and 
            now - SWAGGER_LAST_FETCH < SWAGGER_CACHE_TTL):
            return SWAGGER_CACHE
        
        # Fetch new swagger spec
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RESEARCHBITCOIN_BASE_URL}/v1/swagger.json")
            response.raise_for_status()
            
            swagger_data = response.json()
            SWAGGER_CACHE = swagger_data
            SWAGGER_LAST_FETCH = now
            
            print(f"✓ Successfully fetched swagger specification ({len(swagger_data.get('paths', {}))} paths)")
            return swagger_data
            
    except Exception as e:
        print(f"⚠ Failed to fetch swagger spec: {str(e)}")
        # Return cached version if available
        return SWAGGER_CACHE

def normalize_indicator_name(name: str) -> str:
    """Normalize indicator name for fuzzy matching - lowercase, strip spaces/-/_"""
    return re.sub(r'[_\-\s]+', '', name.lower().strip())

def extract_indicators_from_swagger(swagger_data: Dict) -> Dict[str, IndicatorEntry]:
    """Extract indicator definitions from swagger specification"""
    indicators = {}
    
    if not swagger_data or 'paths' not in swagger_data:
        print("⚠ Invalid swagger data or no paths found")
        return indicators
    
    paths = swagger_data['paths']
    
    for path, path_info in paths.items():
        # Look for paths with data_field parameter patterns
        if '{data_field}' in path:
            
            # Extract category from path (e.g., /supply_distribution/{data_field} -> supply_distribution)
            path_parts = path.strip('/').split('/')
            category = 'general'
            if len(path_parts) >= 1:
                category = path_parts[0]  # First part of the path
            
            # Look for data_field enumerations in parameters
            for method, method_info in path_info.items():
                if method.lower() in ['get', 'post'] and 'parameters' in method_info:
                    for param in method_info['parameters']:
                        if param.get('name') == 'data_field' and 'enum' in param:
                            # Found enumerated data_field values
                            for data_field in param['enum']:
                                # Create indicator entry with full v1 path
                                description = param.get('description', '')
                                if 'schema' in param and 'description' in param['schema']:
                                    description = param['schema']['description']
                                
                                full_endpoint_path = f"/v1{path.replace('{data_field}', data_field)}"
                                
                                indicator = IndicatorEntry(
                                    endpoint_path=full_endpoint_path,
                                    data_field=data_field,
                                    category=category,
                                    description=description
                                )
                                
                                # Add with exact name
                                indicators[data_field] = indicator
                                # Add with normalized name
                                normalized = normalize_indicator_name(data_field)
                                if normalized != data_field:
                                    indicators[normalized] = indicator
    
    return indicators

def build_indicator_index(swagger_data: Optional[Dict] = None) -> None:
    """Build the indicator index from swagger data and aliases"""
    global INDICATOR_INDEX
    
    if swagger_data is None:
        return
    
    # Extract indicators from swagger
    swagger_indicators = extract_indicators_from_swagger(swagger_data)
    
    # Start with swagger indicators
    INDICATOR_INDEX = swagger_indicators.copy()
    
    # Add alias mappings
    for alias, target in INDICATOR_ALIASES.items():
        # Try to find the target in existing indicators
        if target in INDICATOR_INDEX:
            INDICATOR_INDEX[alias] = INDICATOR_INDEX[target]
        else:
            # Create a placeholder entry for the alias
            normalized_target = normalize_indicator_name(target)
            if normalized_target in INDICATOR_INDEX:
                INDICATOR_INDEX[alias] = INDICATOR_INDEX[normalized_target]
    
    print(f"✓ Built indicator index with {len(INDICATOR_INDEX)} entries")

def resolve_indicator(indicator_name: str, visited: Optional[set] = None) -> Optional[IndicatorEntry]:
    """Resolve an indicator name to its entry (with recursion protection)"""
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if indicator_name in visited:
        return None
    visited.add(indicator_name)
    
    # Exact match first
    if indicator_name in INDICATOR_INDEX:
        return INDICATOR_INDEX[indicator_name]
    
    # Normalized match
    normalized = normalize_indicator_name(indicator_name)
    if normalized in INDICATOR_INDEX:
        return INDICATOR_INDEX[normalized]
    
    # Fuzzy match through aliases
    if indicator_name.lower() in INDICATOR_ALIASES:
        target = INDICATOR_ALIASES[indicator_name.lower()]
        if target != indicator_name:  # Avoid self-reference
            return resolve_indicator(target, visited)
    
    return None

def find_similar_indicators(query: str, max_results: int = 50) -> List[Dict[str, str]]:
    """Find indicators similar to the query with improved scoring"""
    query_normalized = normalize_indicator_name(query)
    query_lower = query.lower().strip()
    matches = []
    
    for name, entry in INDICATOR_INDEX.items():
        # Skip normalized duplicates for cleaner results
        if name == entry.data_field:  # Only show primary entries
            score = 0
            name_lower = name.lower()
            name_normalized = normalize_indicator_name(name)
            
            # Exact match (highest priority)
            if query_lower == name_lower:
                score = 100
            # Exact normalized match
            elif query_normalized == name_normalized:
                score = 95
            # Starts with query (prefix match)
            elif name_lower.startswith(query_lower):
                score = 85
            # Starts with normalized query
            elif name_normalized.startswith(query_normalized):
                score = 80
            # Contains query (case-insensitive)
            elif query_lower in name_lower:
                score = 70
            # Contains normalized query
            elif query_normalized in name_normalized:
                score = 65
            # Description match
            elif query_lower in entry.description.lower():
                score = 30
            # Partial word match
            elif any(query_lower in word for word in name_lower.split('_')):
                score = 25
            
            if score > 0:
                matches.append({
                    'name': name,
                    'category': entry.category,
                    'data_field': entry.data_field,
                    'description': entry.description,
                    'score': score
                })
    
    # Sort by score (descending) and return top results
    matches.sort(key=lambda x: (-x['score'], x['name']))  # Secondary sort by name for consistency
    return matches[:max_results]

def get_all_indicators(max_results: int = 200) -> List[Dict[str, str]]:
    """Get all indicators sorted by name"""
    indicators = []
    
    for name, entry in INDICATOR_INDEX.items():
        # Only include primary entries (not normalized duplicates)
        if name == entry.data_field:
            indicators.append({
                'name': name,
                'category': entry.category,
                'data_field': entry.data_field,
                'description': entry.description
            })
    
    # Sort alphabetically by name
    indicators.sort(key=lambda x: x['name'])
    return indicators[:max_results]

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

@app.on_event("startup")
async def startup_event():
    """Initialize the indicator index on startup"""
    print("🚀 Starting Financial Data API Proxy...")
    print("📊 Initializing indicator index...")
    
    try:
        swagger_data = await fetch_swagger_spec()
        if swagger_data:
            build_indicator_index(swagger_data)
            print(f"✅ Indicator index ready with {len(INDICATOR_INDEX)} indicators")
        else:
            print("⚠ Could not fetch swagger spec, using static aliases only")
            build_indicator_index({})  # Build with aliases only
    except Exception as e:
        print(f"❌ Error initializing indicator index: {str(e)}")
        build_indicator_index({})  # Fallback to aliases only

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

@app.post("/tool/get_named_metrics")
async def get_named_metrics(request: Request, named_request: NamedMetricsRequest) -> JSONResponse:
    """
    Get metrics by friendly indicator names with dynamic resolution.
    """
    # Rate limiting
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail={"ok": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded: 30 requests per minute"}}
        )
    
    # Check if token is available
    if not RESEARCHBITCOIN_TOKEN:
        raise HTTPException(
            status_code=502,
            detail={"ok": False, "error": {"code": "CONFIG_ERROR", "message": "ResearchBitcoin API token not configured"}}
        )
    
    # Prepare response structure
    results = {}
    request_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Default date_field to yesterday if not provided
    date_field = named_request.date_field
    if not date_field:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_field = yesterday.strftime("%Y-%m-%d")
    
    # Process each indicator
    for indicator_name in named_request.indicators:
        try:
            # Resolve indicator
            indicator_entry = resolve_indicator(indicator_name)
            
            if not indicator_entry:
                # Provide fuzzy fallback with top 5 suggestions
                suggestions = find_similar_indicators(indicator_name, max_results=5)
                # If no similar found, provide popular indicators as fallback
                if not suggestions:
                    popular_indicators = ["mvrv", "sopr", "supply_coinbase", "mvrv_sth", "sopr_sth"]
                    suggestions = [{"name": name} for name in popular_indicators]
                
                results[indicator_name] = {
                    "error": {
                        "code": "UNKNOWN_INDICATOR",
                        "message": f"Unknown indicator: {indicator_name}",
                        "suggestions": [s['name'] for s in suggestions]
                    }
                }
                continue
            
            # Build query parameters
            query_params = {
                "output_format": "json",
                "date_field": date_field,
                "token": RESEARCHBITCOIN_TOKEN
            }
            
            # Add window parameter if supported and provided
            if named_request.window:
                query_params["window"] = named_request.window
            
            # Construct upstream URL
            upstream_url = f"{RESEARCHBITCOIN_BASE_URL}{indicator_entry.endpoint_path}"
            
            # Make request to upstream API
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(upstream_url, params=query_params)
                
                if response.status_code == 200:
                    try:
                        upstream_data = response.json()
                        results[indicator_name] = {
                            "source": redact_token_from_url(str(response.url)),
                            "data": upstream_data
                        }
                    except Exception as e:
                        results[indicator_name] = {
                            "error": {
                                "code": "UPSTREAM_ERROR",
                                "message": f"Failed to parse upstream JSON: {str(e)}"
                            }
                        }
                else:
                    results[indicator_name] = {
                        "error": {
                            "code": "UPSTREAM_ERROR",
                            "message": f"Upstream API returned {response.status_code}: {response.text[:200]}"
                        }
                    }
                    
        except httpx.TimeoutException:
            results[indicator_name] = {
                "error": {
                    "code": "UPSTREAM_ERROR", 
                    "message": "Upstream API request timed out"
                }
            }
        except Exception as e:
            results[indicator_name] = {
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": f"Unexpected error: {str(e)}"
                }
            }
    
    # Return comprehensive response
    return JSONResponse(
        content={
            "ok": True,
            "as_of": request_timestamp,
            "results": results
        },
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

@app.get("/tool/list_indicators")
async def list_indicators(request: Request, query: Optional[str] = None) -> JSONResponse:
    """
    List available indicators with optional search query.
    Returns top 200 indicators if no query, or up to 50 search results with scoring.
    """
    # Rate limiting
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail={"ok": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded: 30 requests per minute"}}
        )
    
    if query and query.strip():
        # Search for matching indicators (up to 50 results)
        matches = find_similar_indicators(query.strip(), max_results=50)
        return JSONResponse(
            content={
                "ok": True,
                "count": len(matches),
                "items": matches
            },
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    else:
        # Return top 200 indicators when no query provided
        all_indicators = get_all_indicators(max_results=200)
        return JSONResponse(
            content={
                "ok": True,
                "count": len(all_indicators),
                "items": all_indicators
            },
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
            "message": "Financial Data API Proxy",
            "version": "1.0.0",
            "description": "Secure proxy for ResearchBitcoin API",
            "endpoints": {
                "health": "/healthz",
                "status": "/status", 
                "metrics": "/tool/get_metrics",
                "named_metrics": "/tool/get_named_metrics",
                "indicators": "/tool/list_indicators"
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
