"""
Analytics API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime

# Create namespace for analytics API
ns_analytics = Namespace('analytics', description='Analytics operations')

@ns_analytics.route('/performance-report')
class PerformanceReport(Resource):
    @login_required
    def get(self):
        """Get performance report for strategies."""
        try:
            from analytics.performance_tracker import PerformanceTracker
            
            strategy_names = request.args.getlist('strategies')
            lookback_days = int(request.args.get('lookback_days', 90))
            
            tracker = PerformanceTracker()
            report = tracker.generate_performance_report(strategy_names, lookback_days)
            
            return {
                'success': True,
                'performance_report': report,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to get performance report: {str(e)}'}, 500

@ns_analytics.route('/analyze-stock')
class AnalyzeStock(Resource):
    @login_required
    def post(self):
        """Analyze a stock using ChatGPT."""
        try:
            from analysis.chatgpt_analyzer import ChatGPTAnalyzer
            
            data = request.get_json()
            stock_data = data.get('stock_data')
            
            if not stock_data:
                return {'error': 'Stock data required'}, 400
            
            analyzer = ChatGPTAnalyzer()
            analysis = analyzer.analyze_stock(stock_data)
            
            return {
                'success': True,
                'ai_analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to analyze stock: {str(e)}'}, 500

@ns_analytics.route('/analyze-portfolio')
class AnalyzePortfolio(Resource):
    @login_required
    def post(self):
        """Analyze a portfolio using ChatGPT."""
        try:
            from analysis.chatgpt_analyzer import ChatGPTAnalyzer
            
            data = request.get_json()
            suggested_stocks = data.get('suggested_stocks', [])
            strategy_name = data.get('strategy_name', 'Unknown')
            
            if not suggested_stocks:
                return {'error': 'Suggested stocks required'}, 400
            
            analyzer = ChatGPTAnalyzer()
            analysis = analyzer.analyze_portfolio(suggested_stocks, strategy_name)
            
            return {
                'success': True,
                'ai_analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to analyze portfolio: {str(e)}'}, 500