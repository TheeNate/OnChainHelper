# Overview

This project is a Financial Data API built with FastAPI that provides mock financial data endpoints. The application serves as a backend service that returns structured financial metrics and scores, likely designed to support financial analysis or trading applications. The API includes health check functionality and a confluence endpoint that returns mock Bitcoin-related financial indicators such as MVRV-Z scores and SOPR (Spent Output Profit Ratio) metrics.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **FastAPI**: Chosen as the web framework for its automatic API documentation, type hints support, and high performance. FastAPI provides built-in request/response validation and OpenAPI schema generation.

## API Design
- **RESTful endpoints**: The application follows REST principles with clear endpoint naming (`/healthz` for health checks, `/tool/get_confluence` for data retrieval).
- **JSON responses**: All endpoints return structured JSON responses with consistent formatting including status indicators and data payloads.
- **Type hints**: Uses Python type hints for better code documentation and IDE support.

## Cross-Origin Resource Sharing (CORS)
- **Permissive CORS policy**: Currently configured to allow all origins, methods, and headers, making it suitable for development environments but requiring tightening for production use.

## Data Structure
- **Mock data approach**: The application returns hardcoded financial data rather than connecting to live data sources, suggesting this is either a development/testing setup or a prototype.
- **Structured financial metrics**: Returns organized data with scores, components, and state indicators (neutral, bullish) for various financial indicators.

## Error Handling and Monitoring
- **Health check endpoint**: Provides a simple health monitoring capability for deployment and orchestration systems.
- **Consistent response format**: All endpoints follow a similar response structure with `ok` status indicators.

# External Dependencies

## Python Packages
- **FastAPI**: Web framework for building the API
- **uvicorn**: ASGI server for running the FastAPI application
- **typing**: For type annotations and hints

## Runtime Requirements
- **Python 3.7+**: Required for FastAPI compatibility
- **ASGI server**: Uvicorn serves as the application server

## Development and Deployment
- **No database dependencies**: Current implementation uses in-memory mock data
- **No external API integrations**: All data is generated internally
- **No authentication services**: Open API without authentication mechanisms

## Recent Changes (August 12, 2025)

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