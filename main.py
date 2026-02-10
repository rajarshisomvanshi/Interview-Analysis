"""
Interview Intelligence System - Main Entry Point

CLI and server entry point for the system.
"""

import argparse
import uvicorn
import logging

from config import settings

logger = logging.getLogger(__name__)


def run_server():
    """Run the FastAPI server"""
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Interview Intelligence System")
    parser.add_argument(
        "command",
        choices=["server", "analyze"],
        help="Command to run"
    )
    parser.add_argument(
        "--session-id",
        help="Session ID for analysis command"
    )
    
    args = parser.parse_args()
    
    if args.command == "server":
        run_server()
    elif args.command == "analyze":
        if not args.session_id:
            print("Error: --session-id required for analyze command")
            return
        # TODO: Implement CLI analysis
        print(f"Analyzing session: {args.session_id}")


if __name__ == "__main__":
    main()
