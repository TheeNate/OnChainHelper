#!/usr/bin/env python3
"""
Simple startup script for the Financial Data API.
This script ensures the application starts correctly in deployment environments.
"""

import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )