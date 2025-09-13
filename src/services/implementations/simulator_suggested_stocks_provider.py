"""
Simulator Suggested Stocks Provider

Paper trading implementation for suggested stocks functionality.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import random
from ..interfaces.suggested_stocks_interface import ISuggestedStocksProvider, StrategyType, SuggestedStock


class SimulatorSuggestedStocksProvider(ISuggestedStocksProvider):
    """Simulator implementation for suggested stocks."""
    
    def get_suggested_stocks(self, user_id: int, strategies: List[StrategyType] = None, 
                           limit: int = 50) -> Dict[str, Any]:
        """Get simulated suggested stocks."""
        if not strategies:
            strategies = [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]
        
        suggested_stocks = []
        sample_stocks = [
            'NSE:BAJFINANCE-EQ', 'NSE:ASIANPAINT-EQ', 'NSE:MARUTI-EQ', 'NSE:WIPRO-EQ',
            'NSE:LT-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:SUNPHARMA-EQ', 'NSE:HINDUNILVR-EQ'
        ]
        
        for i, symbol in enumerate(sample_stocks[:limit]):
            strategy = strategies[i % len(strategies)]
            price = random.uniform(500, 3000)
            
            stock = SuggestedStock(
                symbol=symbol,
                name=symbol.split(':')[1].replace('-EQ', ''),
                strategy=strategy,
                current_price=price,
                recommendation=random.choice(['BUY', 'HOLD', 'SELL'])
            )
            
            stock.target_price = price * random.uniform(1.05, 1.20)
            stock.stop_loss = price * random.uniform(0.85, 0.95)
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
            stock.reason = f"Selected based on {strategy_name} strategy analysis"
            stock.market_cap = price * random.uniform(100000, 5000000)
            stock.pe_ratio = random.uniform(10, 30)
            stock.pb_ratio = random.uniform(1, 5)
            stock.roe = random.uniform(0.08, 0.25)
            stock.sales_growth = random.uniform(-5, 25)
            
            suggested_stocks.append(stock.to_dict())
        
        return {
            'success': True,
            'data': suggested_stocks,
            'strategies_applied': [s.value if hasattr(s, 'value') else str(s) for s in strategies],
            'last_updated': datetime.now().isoformat()
        }
    
    def get_stock_analysis(self, user_id: int, symbol: str) -> Dict[str, Any]:
        """Get simulated stock analysis."""
        price = random.uniform(500, 3000)
        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'current_price': price,
                'change': random.uniform(-50, 50),
                'change_percent': random.uniform(-5, 5),
                'volume': random.randint(10000, 1000000),
                'high_52w': price * 1.3,
                'low_52w': price * 0.7,
                'market_cap': price * random.uniform(100000, 5000000),
                'pe_ratio': random.uniform(10, 30),
                'pb_ratio': random.uniform(1, 5),
                'dividend_yield': random.uniform(0, 5),
                'roe': random.uniform(8, 25),
                'debt_to_equity': random.uniform(0, 1),
                'recommendation': random.choice(['BUY', 'HOLD', 'SELL']),
                'target_price': price * random.uniform(1.05, 1.20),
                'stop_loss': price * random.uniform(0.85, 0.95)
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_strategy_performance(self, user_id: int, strategy: StrategyType, 
                               period: str = '1M') -> Dict[str, Any]:
        """Get simulated strategy performance."""
        return {
            'success': True,
            'data': {
                'strategy': strategy.value,
                'period': period,
                'total_return': round(random.uniform(-10, 20), 2),
                'win_rate': round(random.uniform(40, 80), 2),
                'avg_return_per_trade': round(random.uniform(-2, 5), 2),
                'max_drawdown': round(random.uniform(-15, -2), 2),
                'sharpe_ratio': round(random.uniform(0.3, 2.0), 2),
                'total_trades': random.randint(10, 50),
                'winning_trades': random.randint(5, 35),
                'losing_trades': random.randint(5, 20)
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_sector_analysis(self, user_id: int) -> Dict[str, Any]:
        """Get simulated sector analysis."""
        sectors = ['Technology', 'Banking', 'Energy', 'Pharmaceuticals', 'FMCG', 'Auto', 'Metals']
        sector_data = []
        
        for sector in sectors:
            sector_data.append({
                'sector': sector,
                'performance': round(random.uniform(-5, 20), 2),
                'recommendation': random.choice(['BUY', 'HOLD', 'SELL']),
                'top_stocks': [f'NSE:STOCK{i}-EQ' for i in range(1, 4)]
            })
        
        return {
            'success': True,
            'data': sector_data,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_technical_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Get simulated technical screener results."""
        stocks = []
        for i in range(random.randint(5, 15)):
            stocks.append({
                'symbol': f'NSE:TECH{i+1}-EQ',
                'name': f'Tech Stock {i+1}',
                'rsi': round(random.uniform(30, 70), 2),
                'macd': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
                'moving_avg_signal': random.choice(['BUY', 'SELL', 'HOLD'])
            })
        
        return {
            'success': True,
            'data': stocks,
            'criteria_applied': criteria,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_fundamental_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Get simulated fundamental screener results."""
        stocks = []
        for i in range(random.randint(5, 15)):
            stocks.append({
                'symbol': f'NSE:FUND{i+1}-EQ',
                'name': f'Fundamental Stock {i+1}',
                'pe_ratio': round(random.uniform(10, 30), 2),
                'roe': round(random.uniform(8, 25), 2),
                'debt_to_equity': round(random.uniform(0, 1), 2),
                'revenue_growth': round(random.uniform(-5, 25), 2)
            })
        
        return {
            'success': True,
            'data': stocks,
            'criteria_applied': criteria,
            'last_updated': datetime.now().isoformat()
        }
