"""
Simulator Portfolio Provider - Paper Trading Implementation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
from ..interfaces.portfolio_interface import IPortfolioProvider, Holding, Position


class SimulatorPortfolioProvider(IPortfolioProvider):
    """Simulator implementation for portfolio management."""
    
    def __init__(self):
        # In-memory storage for simulator portfolio data
        self._holdings = {}
        self._positions = {}
        self._portfolio_history = {}
    
    def get_holdings(self, user_id: int) -> Dict[str, Any]:
        """Get simulated holdings."""
        try:
            # Generate sample holdings if none exist
            if user_id not in self._holdings:
                self._generate_sample_holdings(user_id)
            
            holdings = self._holdings.get(user_id, [])
            
            # Calculate totals
            total_value = sum(holding.get('current_value', 0) for holding in holdings)
            total_pnl = sum(holding.get('pnl', 0) for holding in holdings)
            total_investment = sum(holding.get('investment_value', 0) for holding in holdings)
            
            return {
                'success': True,
                'data': holdings,
                'total_value': round(total_value, 2),
                'total_pnl': round(total_pnl, 2),
                'total_pnl_percent': round((total_pnl / total_investment * 100) if total_investment > 0 else 0, 2),
                'total_investment': round(total_investment, 2),
                'holdings_count': len(holdings),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total_value': 0,
                'total_pnl': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_positions(self, user_id: int) -> Dict[str, Any]:
        """Get simulated positions."""
        try:
            # Generate sample positions if none exist
            if user_id not in self._positions:
                self._generate_sample_positions(user_id)
            
            positions = self._positions.get(user_id, [])
            
            # Calculate totals
            total_pnl = sum(position.get('pnl', 0) for position in positions)
            long_positions = [pos for pos in positions if pos.get('side') == 'long']
            short_positions = [pos for pos in positions if pos.get('side') == 'short']
            
            return {
                'success': True,
                'data': positions,
                'total_pnl': round(total_pnl, 2),
                'long_positions': len(long_positions),
                'short_positions': len(short_positions),
                'total_positions': len(positions),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total_pnl': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """Get simulated portfolio summary."""
        try:
            holdings_response = self.get_holdings(user_id)
            positions_response = self.get_positions(user_id)
            
            if not holdings_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch portfolio data',
                    'data': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings_data = holdings_response.get('data', [])
            positions_data = positions_response.get('data', []) if positions_response.get('success') else []
            
            # Calculate summary metrics
            total_portfolio_value = holdings_response.get('total_value', 0)
            total_pnl = holdings_response.get('total_pnl', 0)
            total_investment = holdings_response.get('total_investment', 0)
            positions_pnl = positions_response.get('total_pnl', 0)
            
            # Simulate available cash
            available_cash = random.uniform(10000, 50000)
            
            # Calculate daily P&L (simulated)
            daily_pnl = random.uniform(-2000, 3000)
            daily_pnl_percent = (daily_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            summary = {
                'total_portfolio_value': round(total_portfolio_value, 2),
                'total_pnl': round(total_pnl + positions_pnl, 2),
                'total_pnl_percent': round(((total_pnl + positions_pnl) / total_investment * 100) if total_investment > 0 else 0, 2),
                'available_cash': round(available_cash, 2),
                'total_balance': round(total_portfolio_value + available_cash, 2),
                'holdings_count': len(holdings_data),
                'positions_count': len(positions_data),
                'daily_pnl': round(daily_pnl, 2),
                'daily_pnl_percent': round(daily_pnl_percent, 2),
                'total_investment': round(total_investment, 2),
                'unrealized_pnl': round(total_pnl, 2),
                'realized_pnl': round(random.uniform(-1000, 2000), 2),
                'margin_used': round(random.uniform(0, 20000), 2),
                'buying_power': round(available_cash * 2, 2)  # Simulated 2x leverage
            }
            
            return {
                'success': True,
                'data': summary,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_allocation(self, user_id: int) -> Dict[str, Any]:
        """Get simulated portfolio allocation."""
        try:
            holdings_response = self.get_holdings(user_id)
            
            if not holdings_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch holdings for allocation analysis',
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings = holdings_response.get('data', [])
            total_value = holdings_response.get('total_value', 0)
            
            # Calculate sector allocation
            sector_allocation = {}
            for holding in holdings:
                sector = holding.get('sector', 'Others')
                value = holding.get('current_value', 0)
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = {
                        'sector': sector,
                        'value': 0,
                        'percentage': 0,
                        'stocks': []
                    }
                
                sector_allocation[sector]['value'] += value
                sector_allocation[sector]['stocks'].append({
                    'symbol': holding.get('symbol', ''),
                    'name': holding.get('name', ''),
                    'value': value
                })
            
            # Calculate percentages and sort
            allocation_list = []
            for sector_data in sector_allocation.values():
                sector_data['percentage'] = round((sector_data['value'] / total_value * 100) if total_value > 0 else 0, 2)
                sector_data['value'] = round(sector_data['value'], 2)
                allocation_list.append(sector_data)
            
            allocation_list.sort(key=lambda x: x['value'], reverse=True)
            
            return {
                'success': True,
                'data': allocation_list,
                'allocation_type': 'sector',
                'total_portfolio_value': total_value,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_performance(self, user_id: int, period: str = '1M') -> Dict[str, Any]:
        """Get simulated portfolio performance."""
        try:
            summary_response = self.get_portfolio_summary(user_id)
            
            if not summary_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch portfolio data for performance analysis',
                    'data': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            summary_data = summary_response.get('data', {})
            
            # Simulate performance metrics
            period_multiplier = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}
            days = period_multiplier.get(period, 30)
            
            # Simulate returns based on period
            base_return = random.uniform(-5, 15)
            period_return = base_return * (days / 30)  # Scale by period
            
            performance_data = {
                'period': period,
                'portfolio_return': round(period_return, 2),
                'portfolio_value': summary_data.get('total_portfolio_value', 0),
                'total_pnl': summary_data.get('total_pnl', 0),
                'best_performing_stock': {
                    'symbol': 'NSE:WINNER-EQ',
                    'name': 'Best Performer',
                    'return_percent': round(random.uniform(10, 30), 2)
                },
                'worst_performing_stock': {
                    'symbol': 'NSE:LOSER-EQ',
                    'name': 'Worst Performer',
                    'return_percent': round(random.uniform(-20, -5), 2)
                },
                'volatility': round(random.uniform(10, 25), 2),
                'sharpe_ratio': round(random.uniform(0.5, 2.0), 2),
                'max_drawdown': round(random.uniform(-15, -2), 2),
                'win_rate': round(random.uniform(45, 75), 2),
                'risk_adjusted_return': round(period_return / 15, 2),  # Simplified
                'benchmark_comparison': {
                    'benchmark_name': 'NIFTY 50',
                    'benchmark_return': round(random.uniform(-3, 12), 2),
                    'relative_performance': round(period_return - random.uniform(-3, 12), 2),
                    'beta': round(random.uniform(0.8, 1.3), 2),
                    'alpha': round(random.uniform(-2, 5), 2)
                }
            }
            
            return {
                'success': True,
                'data': performance_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def get_dividend_history(self, user_id: int, start_date: datetime = None, 
                           end_date: datetime = None) -> Dict[str, Any]:
        """Get simulated dividend history."""
        try:
            holdings_response = self.get_holdings(user_id)
            
            if not holdings_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch holdings for dividend analysis',
                    'data': [],
                    'total_dividends': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings = holdings_response.get('data', [])
            
            # Generate simulated dividend records
            dividend_records = []
            total_dividends = 0
            
            for holding in holdings:
                if holding.get('current_value', 0) > 10000:  # Only larger holdings pay dividends
                    dividend_amount = holding.get('current_value', 0) * random.uniform(0.01, 0.04)  # 1-4% yield
                    dividend_records.append({
                        'symbol': holding.get('symbol', ''),
                        'symbol_name': holding.get('name', ''),
                        'dividend_amount': round(dividend_amount, 2),
                        'dividend_yield': round(random.uniform(1, 4), 2),
                        'ex_date': (datetime.now() - timedelta(days=random.randint(10, 60))).strftime('%Y-%m-%d'),
                        'pay_date': (datetime.now() - timedelta(days=random.randint(5, 30))).strftime('%Y-%m-%d'),
                        'quantity_held': holding.get('quantity', 0)
                    })
                    total_dividends += dividend_amount
            
            return {
                'success': True,
                'data': dividend_records,
                'total_dividends': round(total_dividends, 2),
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
                    'end_date': end_date.strftime('%Y-%m-%d') if end_date else None
                },
                'note': 'Simulated dividend data for testing purposes',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total_dividends': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_risk_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get simulated portfolio risk metrics."""
        try:
            holdings_response = self.get_holdings(user_id)
            positions_response = self.get_positions(user_id)
            
            if not holdings_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch portfolio data for risk analysis',
                    'data': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings = holdings_response.get('data', [])
            positions = positions_response.get('data', []) if positions_response.get('success') else []
            
            # Simulate risk metrics
            risk_metrics = {
                'portfolio_beta': round(random.uniform(0.8, 1.4), 2),
                'value_at_risk': {
                    'var_95': round(random.uniform(5000, 15000), 2),
                    'var_99': round(random.uniform(8000, 25000), 2)
                },
                'concentration_risk': {
                    'risk_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'largest_position': round(random.uniform(5, 25), 2),
                    'top_5_concentration': round(random.uniform(30, 70), 2),
                    'number_of_holdings': len(holdings)
                },
                'sector_risk': {
                    'risk_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'max_sector_allocation': round(random.uniform(15, 45), 2),
                    'sector_count': len(set(h.get('sector', 'Others') for h in holdings))
                },
                'correlation_risk': round(random.uniform(0.3, 0.8), 2),
                'leverage_ratio': round(random.uniform(1.0, 2.5), 2),
                'max_position_size': round(random.uniform(5, 20), 2),
                'diversification_ratio': round(random.uniform(0.6, 0.9), 2),
                'risk_rating': random.choice(['LOW', 'MEDIUM', 'HIGH'])
            }
            
            # Generate risk recommendations
            recommendations = [
                "Consider diversifying across more sectors to reduce concentration risk",
                "Monitor your largest position size to maintain balanced allocation",
                "Review your portfolio regularly to ensure risk levels remain acceptable"
            ]
            
            return {
                'success': True,
                'data': {
                    **risk_metrics,
                    'recommendations': recommendations,
                    'risk_summary': f"Your portfolio has {risk_metrics['risk_rating'].lower()} risk levels. Monitor regularly for changes."
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def _generate_sample_holdings(self, user_id: int):
        """Generate sample holdings for simulation."""
        sample_stocks = [
            {'symbol': 'NSE:RELIANCE-EQ', 'name': 'Reliance Industries', 'sector': 'Energy'},
            {'symbol': 'NSE:TCS-EQ', 'name': 'Tata Consultancy Services', 'sector': 'Technology'},
            {'symbol': 'NSE:HDFCBANK-EQ', 'name': 'HDFC Bank', 'sector': 'Banking'},
            {'symbol': 'NSE:INFY-EQ', 'name': 'Infosys', 'sector': 'Technology'},
            {'symbol': 'NSE:ICICIBANK-EQ', 'name': 'ICICI Bank', 'sector': 'Banking'},
            {'symbol': 'NSE:ITC-EQ', 'name': 'ITC Limited', 'sector': 'FMCG'},
            {'symbol': 'NSE:SBIN-EQ', 'name': 'State Bank of India', 'sector': 'Banking'},
            {'symbol': 'NSE:BHARTIARTL-EQ', 'name': 'Bharti Airtel', 'sector': 'Telecom'}
        ]
        
        holdings = []
        for i, stock in enumerate(sample_stocks[:random.randint(3, 6)]):
            quantity = random.randint(5, 50)
            avg_price = random.uniform(100, 3000)
            current_price = avg_price * random.uniform(0.85, 1.25)  # ±25% from avg price
            
            holding = Holding(
                symbol=stock['symbol'],
                name=stock['name'],
                quantity=quantity,
                avg_price=avg_price,
                current_price=current_price
            )
            
            # Add additional fields
            holding_dict = holding.to_dict()
            holding_dict['sector'] = stock['sector']
            holding_dict['exchange'] = 'NSE'
            holding_dict['product'] = 'EQ'
            
            holdings.append(holding_dict)
        
        self._holdings[user_id] = holdings
    
    def _generate_sample_positions(self, user_id: int):
        """Generate sample positions for simulation."""
        sample_symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ']
        positions = []
        
        for i in range(random.randint(0, 3)):
            symbol = random.choice(sample_symbols)
            side = random.choice(['long', 'short'])
            quantity = random.randint(10, 100)
            avg_price = random.uniform(100, 3000)
            current_price = avg_price * random.uniform(0.9, 1.15)
            
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                avg_price=avg_price,
                current_price=current_price
            )
            
            # Add additional fields
            position_dict = position.to_dict()
            position_dict['product_type'] = 'INTRADAY'
            position_dict['exchange'] = 'NSE'
            position_dict['day_change'] = random.uniform(-1000, 2000)
            
            positions.append(position_dict)
        
        self._positions[user_id] = positions
