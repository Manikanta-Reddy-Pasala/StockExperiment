"""
Simplified main application entry point
"""
from src.web.app import create_app
from config import DEBUG, HOST, PORT

def main():
    """Main application entry point."""
    app = create_app()
    
    print(f"Starting trading system on {HOST}:{PORT}")
    print(f"Debug mode: {DEBUG}")
    
    app.run(
        debug=DEBUG,
        host=HOST,
        port=PORT
    )

if __name__ == "__main__":
    main()
