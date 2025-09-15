"""
Request Logger Middleware

Logs all incoming requests and outgoing responses for debugging API issues.
"""

import json
import time
from datetime import datetime
from flask import request, g
from werkzeug.exceptions import HTTPException


class RequestLoggerMiddleware:
    """Middleware to log all requests and responses."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the middleware with Flask app."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.errorhandler(Exception)(self.log_exception)
    
    def before_request(self):
        """Log incoming request details."""
        g.start_time = time.time()
        g.request_id = str(int(time.time() * 1000))[-8:]  # Simple request ID
        
        # Skip logging for static files
        if request.endpoint and 'static' in request.endpoint:
            return
        
        request_data = {
            'request_id': g.request_id,
            'method': request.method,
            'url': request.url,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'content_type': request.headers.get('Content-Type', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add query parameters
        if request.args:
            request_data['query_params'] = dict(request.args)
        
        # Add form data
        if request.form:
            request_data['form_data'] = dict(request.form)
        
        # Add JSON data
        if request.is_json:
            try:
                request_data['json_data'] = request.get_json()
            except Exception:
                request_data['json_data'] = 'Invalid JSON'
        
        # Print to console with clear formatting
        print("\n" + "="*80)
        print(f"🔵 INCOMING REQUEST [{g.request_id}]")
        print("="*80)
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Path: {request.path}")
        print(f"Remote Address: {request.remote_addr}")
        print(f"Timestamp: {request_data['timestamp']}")
        
        if request.args:
            print(f"Query Params: {dict(request.args)}")
        
        if request.form:
            print(f"Form Data: {dict(request.form)}")
        
        if request.is_json:
            try:
                json_data = request.get_json()
                print(f"JSON Data: {json.dumps(json_data, indent=2)}")
            except Exception:
                print("JSON Data: Invalid JSON")
        
        print("="*80)
    
    def after_request(self, response):
        """Log response details."""
        # Skip logging for static files
        if request.endpoint and 'static' in request.endpoint:
            return response
        
        duration = (time.time() - g.start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            'request_id': getattr(g, 'request_id', 'unknown'),
            'status_code': response.status_code,
            'content_type': response.headers.get('Content-Type', ''),
            'content_length': response.headers.get('Content-Length', ''),
            'duration_ms': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get response data safely
        response_content = None
        try:
            if response.is_json:
                try:
                    response_content = response.get_json()
                except Exception:
                    response_content = 'Invalid JSON response'
            elif hasattr(response, 'data') and response.data:
                # Check if response is in direct passthrough mode
                try:
                    # Try to access data length first to detect passthrough mode
                    data_length = len(response.data)
                    if data_length < 1000:  # Only log small responses
                        response_content = response.data.decode('utf-8')
                    else:
                        response_content = f'Large response ({data_length} bytes)'
                except RuntimeError as e:
                    if "direct passthrough mode" in str(e):
                        response_content = 'Response data not accessible (direct passthrough mode)'
                    else:
                        response_content = f'Data access error: {str(e)}'
                except Exception:
                    response_content = 'Binary data'
        except Exception as e:
            response_content = f'Response logging error: {str(e)}'
        
        # Print to console with clear formatting
        print("\n" + "="*80)
        print(f"🟢 OUTGOING RESPONSE [{getattr(g, 'request_id', 'unknown')}]")
        print("="*80)
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('Content-Type', '')}")
        print(f"Duration: {round(duration, 2)}ms")
        print(f"Timestamp: {response_data['timestamp']}")
        
        if response_content is not None:
            if isinstance(response_content, dict):
                print(f"Response Data: {json.dumps(response_content, indent=2)}")
            else:
                print(f"Response Data: {response_content}")
        
        # Special handling for error responses
        if response.status_code >= 400:
            print(f"⚠️  ERROR RESPONSE: {response.status_code}")
            if response_content:
                print(f"Error Details: {response_content}")
        
        print("="*80)
        
        return response
    
    def log_exception(self, error):
        """Log exceptions."""
        request_id = getattr(g, 'request_id', 'unknown')
        duration = (time.time() - getattr(g, 'start_time', time.time())) * 1000
        
        print("\n" + "="*80)
        print(f"🔴 EXCEPTION [{request_id}]")
        print("="*80)
        print(f"Exception Type: {type(error).__name__}")
        print(f"Exception Message: {str(error)}")
        print(f"Request Method: {request.method}")
        print(f"Request URL: {request.url}")
        print(f"Duration: {round(duration, 2)}ms")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Print traceback for debugging
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("="*80)
        
        # Re-raise the exception
        if isinstance(error, HTTPException):
            return error
        raise error
