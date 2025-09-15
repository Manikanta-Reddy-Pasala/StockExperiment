    # Demo/Fallback API Routes (No Authentication Required)
    # These routes provide mock data for demonstration purposes
    
    @app.route('/api/demo/market/overview', methods=['GET'])
    def api_demo_market_overview():
        """Get demo market overview data (no auth required)."""
        try:
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_market_overview())
        except Exception as e:
            app.logger.error(f"Error getting demo market overview: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable',
                'data': []
            }), 500
    
    @app.route('/api/demo/dashboard/metrics', methods=['GET'])
    def api_demo_dashboard_metrics():
        """Get demo dashboard metrics (no auth required)."""
        try:
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_portfolio_summary())
        except Exception as e:
            app.logger.error(f"Error getting demo dashboard metrics: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500
    
    @app.route('/api/demo/dashboard/portfolio-holdings', methods=['GET'])
    def api_demo_portfolio_holdings():
        """Get demo portfolio holdings (no auth required)."""
        try:
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_portfolio_holdings())
        except Exception as e:
            app.logger.error(f"Error getting demo portfolio holdings: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500
    
    @app.route('/api/demo/dashboard/recent-orders', methods=['GET'])
    def api_demo_recent_orders():
        """Get demo recent orders (no auth required)."""
        try:
            limit = request.args.get('limit', 5, type=int)
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_recent_orders(limit))
        except Exception as e:
            app.logger.error(f"Error getting demo recent orders: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500
    
    @app.route('/api/demo/dashboard/pending-orders', methods=['GET'])
    def api_demo_pending_orders():
        """Get demo pending orders (no auth required)."""
        try:
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_pending_orders())
        except Exception as e:
            app.logger.error(f"Error getting demo pending orders: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500
    
    @app.route('/api/demo/dashboard/portfolio-performance', methods=['GET'])
    def api_demo_portfolio_performance():
        """Get demo portfolio performance (no auth required)."""
        try:
            period = request.args.get('period', '1W')
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_portfolio_performance(period))
        except Exception as e:
            app.logger.error(f"Error getting demo portfolio performance: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500
    
    @app.route('/api/demo/suggested-stocks', methods=['GET'])
    def api_demo_suggested_stocks():
        """Get demo suggested stocks (no auth required)."""
        try:
            from ..services.mock_data_service import get_mock_data_service
            mock_service = get_mock_data_service()
            return jsonify(mock_service.get_suggested_stocks())
        except Exception as e:
            app.logger.error(f"Error getting demo suggested stocks: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Demo data unavailable'
            }), 500

    # Enhanced API Routes with Fallback to Mock Data
    # These routes try real data first, then fall back to mock data
    
    @app.route('/api/market/overview', methods=['GET'])
    def api_market_overview_with_fallback():
        """Get market overview with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_market_overview()
            else:
                # Fall back to demo data
                return api_demo_market_overview()
        except Exception as e:
            app.logger.error(f"Error in market overview with fallback: {str(e)}")
            return api_demo_market_overview()
    
    @app.route('/api/dashboard/metrics', methods=['GET'])
    def api_dashboard_metrics_with_fallback():
        """Get dashboard metrics with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_dashboard_metrics()
            else:
                # Fall back to demo data
                return api_demo_dashboard_metrics()
        except Exception as e:
            app.logger.error(f"Error in dashboard metrics with fallback: {str(e)}")
            return api_demo_dashboard_metrics()
    
    @app.route('/api/dashboard/portfolio-holdings', methods=['GET'])
    def api_portfolio_holdings_with_fallback():
        """Get portfolio holdings with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_portfolio_holdings()
            else:
                # Fall back to demo data
                return api_demo_portfolio_holdings()
        except Exception as e:
            app.logger.error(f"Error in portfolio holdings with fallback: {str(e)}")
            return api_demo_portfolio_holdings()
    
    @app.route('/api/dashboard/recent-orders', methods=['GET'])
    def api_recent_orders_with_fallback():
        """Get recent orders with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_recent_orders()
            else:
                # Fall back to demo data
                return api_demo_recent_orders()
        except Exception as e:
            app.logger.error(f"Error in recent orders with fallback: {str(e)}")
            return api_demo_recent_orders()
    
    @app.route('/api/dashboard/pending-orders', methods=['GET'])
    def api_pending_orders_with_fallback():
        """Get pending orders with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_pending_orders()
            else:
                # Fall back to demo data
                return api_demo_pending_orders()
        except Exception as e:
            app.logger.error(f"Error in pending orders with fallback: {str(e)}")
            return api_demo_pending_orders()
    
    @app.route('/api/dashboard/portfolio-performance', methods=['GET'])
    def api_portfolio_performance_with_fallback():
        """Get portfolio performance with fallback to mock data."""
        try:
            # Try authenticated route first
            if current_user and current_user.is_authenticated:
                return api_get_portfolio_performance()
            else:
                # Fall back to demo data
                return api_demo_portfolio_performance()
        except Exception as e:
            app.logger.error(f"Error in portfolio performance with fallback: {str(e)}")
            return api_demo_portfolio_performance()

