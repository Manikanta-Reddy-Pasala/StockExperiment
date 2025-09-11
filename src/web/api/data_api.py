"""
Data API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime
from data.manager import get_data_provider_manager

# Create namespace for data API
ns_data = Namespace('data', description='Data operations')

@ns_data.route('/stock-data')
class StockData(Resource):
    @login_required
    def post(self):
        """Get stock data from multiple sources including FYERS API."""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return {'error': 'Symbol required'}, 400
            
            # Get FYERS connector from app context if available
            from flask import current_app
            fyers_connector = getattr(current_app, 'fyers_connector', None)
            provider_manager = get_data_provider_manager(fyers_connector)
            stock_data = provider_manager.get_stock_data(symbol)
            
            if not stock_data:
                return {'error': f'No data available for {symbol}'}, 404
            
            return {
                'success': True,
                'stock_data': stock_data,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to get stock data: {str(e)}'}, 500

@ns_data.route('/current-price')
class CurrentPrice(Resource):
    @login_required
    def post(self):
        """Get current price for a stock."""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return {'error': 'Symbol required'}, 400
            
            # Get FYERS connector from app context if available
            from flask import current_app
            fyers_connector = getattr(current_app, 'fyers_connector', None)
            provider_manager = get_data_provider_manager(fyers_connector)
            price = provider_manager.get_current_price(symbol)
            
            if price is None:
                return {'error': f'No price data available for {symbol}'}, 404
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': price,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to get current price: {str(e)}'}, 500

@ns_data.route('/providers')
class DataProviders(Resource):
    @login_required
    def get(self):
        """Get list of available data providers."""
        try:
            # Get FYERS connector from app context if available
            from flask import current_app
            fyers_connector = getattr(current_app, 'fyers_connector', None)
            provider_manager = get_data_provider_manager(fyers_connector)
            providers = provider_manager.get_available_providers()
            
            return {
                'success': True,
                'providers': providers,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to get data providers: {str(e)}'}, 500