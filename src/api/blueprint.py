"""
Main API Blueprint
"""
from flask import Blueprint
from api.v1.admin import admin_bp
from api.v1.trading import trading_bp
from api.common.errors import handle_api_error

# Create main API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Register version 1 blueprints
api_bp.register_blueprint(admin_bp, url_prefix='/v1')
api_bp.register_blueprint(trading_bp, url_prefix='/v1')

# Register error handlers
api_bp.errorhandler(Exception)(handle_api_error)

# API Documentation endpoint
@api_bp.route('/docs')
def api_docs():
    """API Documentation endpoint."""
    from flask import render_template_string
    import os
    
    # Read the API documentation
    docs_path = os.path.join(os.path.dirname(__file__), 'docs', 'api_documentation.md')
    
    try:
        with open(docs_path, 'r') as f:
            docs_content = f.read()
        
        # Convert markdown to HTML (basic conversion)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System API Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .endpoint {{ background: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }}
            </style>
        </head>
        <body>
            <h1>Trading System API Documentation</h1>
            <div class="endpoint">
                <h2>Quick Start</h2>
                <p><strong>Base URL:</strong> <code>http://localhost:5001/api/v1</code></p>
                <p><strong>Authentication:</strong> All endpoints require login via Flask-Login session</p>
                <p><strong>Response Format:</strong> All responses include success/error status and timestamp</p>
            </div>
            <pre>{docs_content}</pre>
        </body>
        </html>
        """
        
        return html_content
    except Exception as e:
        return f"Error loading documentation: {str(e)}", 500

# Health check endpoint
@api_bp.route('/health')
def api_health():
    """API health check."""
    from flask import jsonify
    from datetime import datetime
    
    return jsonify({
        'status': 'healthy',
        'api_version': 'v1',
        'timestamp': datetime.utcnow().isoformat(),
        'endpoints': {
            'admin': '/api/v1/admin',
            'trading': '/api/v1/trading',
            'docs': '/api/docs'
        }
    })
