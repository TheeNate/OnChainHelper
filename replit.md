# Overview

This project is a Financial Data API Proxy built with FastAPI that serves as a secure proxy for the ResearchBitcoin API. The application provides real Bitcoin metrics and financial data by securely forwarding requests to the ResearchBitcoin API while adding authentication, rate limiting, and security validation. The proxy is designed to enable ChatGPT Actions and other authorized clients to access real Bitcoin data without exposing API tokens.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **FastAPI**: Chosen as the web framework for its automatic API documentation, type hints support, and high performance. FastAPI provides built-in request/response validation and OpenAPI schema generation.

## API Design
- **Secure proxy architecture**: The application acts as a secure intermediary between clients and the ResearchBitcoin API
- **Primary endpoint**: `/tool/get_metrics` - Accepts endpoint_path and query parameters, forwards to upstream API with injected authentication
- **RESTful design**: Health checks at `/healthz` and `/status`, main proxy functionality at `/tool/get_metrics`
- **JSON responses**: All endpoints return structured JSON with consistent `ok` status indicators and error handling
- **Type validation**: Uses Pydantic models for request validation and type safety

## Cross-Origin Resource Sharing (CORS)
- **Restrictive CORS policy**: Configured to allow specific origins including ChatGPT domains, localhost for testing, and Replit domains for development.
- **Production security**: CORS is properly configured for production use with trusted origins only.

## Security Features
- **Rate limiting**: 30 requests per minute per IP address to prevent abuse
- **Path validation**: Strict validation of endpoint paths to prevent path traversal attacks and ensure only valid ResearchBitcoin API paths are accessed
- **Token security**: API tokens are read from environment variables and never accepted from clients or logged in responses
- **Input sanitization**: All input parameters are validated using Pydantic models with proper type checking

## Data Flow
- **Real-time data**: The application forwards requests to the live ResearchBitcoin API and returns authentic Bitcoin metrics
- **Secure proxy pattern**: Clients make requests to the proxy, which adds authentication and forwards to the upstream API
- **Error handling**: Comprehensive error handling for upstream API failures, timeouts, and invalid responses

## Error Handling and Monitoring
- **Health check endpoint**: Provides a simple health monitoring capability for deployment and orchestration systems.
- **Consistent response format**: All endpoints follow a similar response structure with `ok` status indicators.

# External Dependencies

## Python Packages
- **FastAPI**: Web framework for building the API with automatic OpenAPI documentation
- **uvicorn**: ASGI server for running the FastAPI application in production
- **httpx**: Async HTTP client for making requests to the ResearchBitcoin API
- **pydantic**: Data validation and serialization using Python type hints
- **typing**: Enhanced type annotations and hints for better code safety

## Runtime Requirements
- **Python 3.7+**: Required for FastAPI compatibility
- **ASGI server**: Uvicorn serves as the application server

## Development and Deployment
- **No database dependencies**: Uses in-memory rate limiting storage (suitable for single-instance deployment)
- **External API integration**: Secure integration with ResearchBitcoin API for real Bitcoin metrics
- **Environment-based authentication**: API tokens managed through secure environment variables
- **Production-ready**: Configured for deployment with proper error handling, timeouts, and security measures

## Recent Changes (August 12, 2025)

### Dynamic Indicator Resolution System - Latest Major Feature
- **Comprehensive indicator discovery**: Implemented automatic swagger spec fetching and parsing to build dynamic indicator index with 617 entries across multiple categories
- **Friendly name resolution**: Added intelligent indicator resolution with exact, normalized, and alias matching for natural language requests
- **New endpoint: `/tool/get_named_metrics`**: 
  - Accepts friendly indicator names like "sth_sopr", "short_term_holder_sopr", "coinbase_supply"
  - Automatically resolves to correct API endpoints and data fields
  - Returns structured response with real data for each requested indicator
  - Handles errors gracefully with helpful suggestions for unknown indicators
- **New endpoint: `/tool/list_indicators`**: 
  - Lists all available indicators grouped by category
  - Provides search functionality with scoring and ranking
  - Returns 308 unique indicators with descriptions and metadata
- **Intelligent alias system**: Pre-configured aliases for common Bitcoin metrics:
  - SOPR variants: "sth_sopr" → "sopr_sth", "lth_sopr" → "sopr_lth"
  - Supply metrics: "coinbase_supply" → "supply_coinbase", "circulating_supply" → "supply_circulating"
  - Market indicators: "mvrv" → "mvrv", with category-based resolution
- **Automatic swagger caching**: 24-hour cache with automatic refresh, fallback to last good copy on fetch failures
- **Production-ready error handling**: Comprehensive error codes, timeout protection, recursion prevention
- **Security maintained**: All rate limiting, token security, and validation features preserved

## Recent Changes (August 12, 2025)

### API Transformation - Secure ResearchBitcoin Proxy Implementation
- **Complete architecture transformation**: Converted from mock data API to secure proxy for ResearchBitcoin API
- **New primary endpoint**: Implemented `/tool/get_metrics` that securely proxies requests to https://api.researchbitcoin.net
- **Authentication integration**: Added secure token injection from RESEARCHBITCOIN_TOKEN environment variable
- **Security hardening**: 
  - Rate limiting: 30 requests per minute per IP
  - Path validation: Prevents path traversal and ensures only valid API paths
  - Input sanitization: Strict validation of all parameters
  - Token security: Never logs or exposes API tokens
- **CORS restriction**: Updated from permissive to restrictive CORS policy allowing only ChatGPT domains and development origins
- **Real data integration**: Successfully tested with live ResearchBitcoin API returning authentic Bitcoin supply distribution data
- **Error handling**: Comprehensive error handling for upstream API failures, timeouts, and validation errors
- **Added dependencies**: httpx for async HTTP requests, enhanced pydantic validation
- **Production ready**: 10-second timeout, proper HTTP status codes, structured error responses

### Deployment Configuration Fixes - Latest Update
- **Applied comprehensive deployment fixes**: Resolved deployment failures that were caused by undefined $file variable in run command:
  - **Enhanced port configuration**: Modified main.py to read PORT from environment variables for deployment flexibility
  - **Created run.py startup script**: Added alternative startup script with environment-aware port configuration
  - **Updated pyproject.toml**: Enhanced project metadata and added start script entry point
  - **Confirmed production settings**: Verified reload=True was already removed for production compatibility
  - **Health check verified**: Confirmed that `/healthz` endpoint returns proper 200 OK responses

### Earlier Deployment Configuration Fixes
- **Removed reload=True from uvicorn configuration**: Fixed production deployment compatibility by removing the development-only reload parameter that was causing deployment failures
- **Verified application startup**: Confirmed that the application starts correctly and health checks pass
- **Production-ready server configuration**: Updated server configuration to be suitable for cloud deployment environments

Note: The application is now fully deployment-ready with comprehensive production configurations. Health checks pass successfully and the application properly handles port configuration from environment variables.