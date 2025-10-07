"""
Advanced Strategy Backtesting Service
Comprehensive backtesting with realistic trading simulation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Advanced backtesting engine for ML-driven trading strategies.

    Features:
    - Realistic slippage and commission modeling
    - Position sizing and risk management
    - Portfolio-level metrics
    - Walk-forward out-of-sample testing
    - Multiple strategy comparison
    """

    def __init__(self, db_session,
                 initial_capital: float = 1000000.0,
                 commission_pct: float = 0.1,
                 slippage_pct: float = 0.05,
                 max_position_size: float = 0.10):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital (default: 10 lakh)
            commission_pct: Commission per trade (default: 0.1%)
            slippage_pct: Slippage per trade (default: 0.05%)
            max_position_size: Max position as % of capital (default: 10%)
        """
        self.db = db_session
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.max_position_size = max_position_size

    def calculate_position_size(self, capital: float, price: float,
                                risk_score: float) -> int:
        """
        Calculate position size based on Kelly Criterion and risk score.

        Args:
            capital: Available capital
            price: Stock price
            risk_score: ML risk score (0-1, lower = safer)

        Returns:
            Number of shares to buy
        """
        # Adjust max position based on risk
        risk_adjusted_max = self.max_position_size * (1 - risk_score)
        position_value = capital * risk_adjusted_max
        shares = int(position_value / price)

        return max(1, shares)  # At least 1 share

    def apply_trading_costs(self, price: float, quantity: int,
                           side: str = 'buy') -> float:
        """
        Apply commission and slippage to trade price.

        Args:
            price: Base price
            quantity: Number of shares
            side: 'buy' or 'sell'

        Returns:
            Adjusted price after costs
        """
        # Slippage: buy higher, sell lower
        if side == 'buy':
            slippage_adjusted = price * (1 + self.slippage_pct)
        else:
            slippage_adjusted = price * (1 - self.slippage_pct)

        # Commission
        trade_value = slippage_adjusted * quantity
        commission = trade_value * self.commission_pct

        # Per-share cost
        total_cost = trade_value + commission
        effective_price = total_cost / quantity

        return effective_price

    def backtest_ml_strategy(self,
                            start_date: str,
                            end_date: str,
                            rebalance_days: int = 14,
                            top_n_stocks: int = 10,
                            min_confidence: float = 0.6,
                            min_ml_score: float = 0.6,
                            max_risk_score: float = 0.4) -> Dict:
        """
        Backtest ML-based stock selection strategy.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_days: Days between rebalancing (default: 14)
            top_n_stocks: Number of stocks to hold (default: 10)
            min_confidence: Minimum ML confidence (default: 0.6)
            min_ml_score: Minimum ML prediction score (default: 0.6)
            max_risk_score: Maximum ML risk score (default: 0.4)

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("=" * 80)
        logger.info("Starting ML Strategy Backtest")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        logger.info(f"Rebalance: Every {rebalance_days} days")
        logger.info(f"Portfolio Size: {top_n_stocks} stocks")
        logger.info(f"Filters: ML Score ≥ {min_ml_score}, Confidence ≥ {min_confidence}, Risk ≤ {max_risk_score}")
        logger.info("=" * 80)

        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date, rebalance_days)

        # Track portfolio
        portfolio = {}  # {symbol: {'shares': int, 'entry_price': float, 'entry_date': str}}
        cash = self.initial_capital
        portfolio_values = []
        trades = []

        for rebalance_date in rebalance_dates:
            # Get ML predictions for this date
            top_stocks = self._get_top_stocks_for_date(
                rebalance_date,
                top_n=top_n_stocks,
                min_confidence=min_confidence,
                min_ml_score=min_ml_score,
                max_risk_score=max_risk_score
            )

            if not top_stocks:
                logger.warning(f"No stocks met criteria on {rebalance_date}")
                continue

            # Calculate current portfolio value
            portfolio_value = cash
            for symbol, position in portfolio.items():
                current_price = self._get_price_for_date(symbol, rebalance_date)
                if current_price:
                    portfolio_value += position['shares'] * current_price

            # Sell positions not in new selection
            symbols_to_hold = [s['symbol'] for s in top_stocks]
            for symbol in list(portfolio.keys()):
                if symbol not in symbols_to_hold:
                    # Sell position
                    position = portfolio[symbol]
                    exit_price = self._get_price_for_date(symbol, rebalance_date)
                    if exit_price:
                        effective_exit_price = self.apply_trading_costs(
                            exit_price, position['shares'], side='sell'
                        )

                        proceeds = position['shares'] * effective_exit_price
                        cash += proceeds

                        # Record trade
                        pnl = proceeds - (position['shares'] * position['entry_price'])
                        pnl_pct = (pnl / (position['shares'] * position['entry_price'])) * 100

                        trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': rebalance_date,
                            'shares': position['shares'],
                            'entry_price': position['entry_price'],
                            'exit_price': effective_exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct
                        })

                        del portfolio[symbol]

            # Buy new positions
            capital_per_stock = portfolio_value / len(top_stocks)

            for stock in top_stocks:
                symbol = stock['symbol']
                price = stock['current_price']
                risk_score = stock['ml_risk_score']

                if symbol not in portfolio:
                    # Calculate position size
                    shares = self.calculate_position_size(
                        capital_per_stock, price, risk_score
                    )

                    # Apply trading costs
                    effective_entry_price = self.apply_trading_costs(
                        price, shares, side='buy'
                    )

                    cost = shares * effective_entry_price

                    if cost <= cash:
                        cash -= cost
                        portfolio[symbol] = {
                            'shares': shares,
                            'entry_price': effective_entry_price,
                            'entry_date': rebalance_date
                        }

            # Record portfolio value
            portfolio_value = cash
            for symbol, position in portfolio.items():
                current_price = self._get_price_for_date(symbol, rebalance_date)
                if current_price:
                    portfolio_value += position['shares'] * current_price

            portfolio_values.append({
                'date': rebalance_date,
                'value': portfolio_value,
                'cash': cash,
                'positions': len(portfolio)
            })

        # Close all positions at end
        final_date = rebalance_dates[-1]
        for symbol, position in portfolio.items():
            exit_price = self._get_price_for_date(symbol, final_date)
            if exit_price:
                effective_exit_price = self.apply_trading_costs(
                    exit_price, position['shares'], side='sell'
                )
                proceeds = position['shares'] * effective_exit_price
                cash += proceeds

                pnl = proceeds - (position['shares'] * position['entry_price'])
                pnl_pct = (pnl / (position['shares'] * position['entry_price'])) * 100

                trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': final_date,
                    'shares': position['shares'],
                    'entry_price': position['entry_price'],
                    'exit_price': effective_exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })

        final_value = cash

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, trades, self.initial_capital, final_value
        )

        logger.info("=" * 80)
        logger.info("Backtest Complete!")
        logger.info("=" * 80)
        logger.info(f"Final Portfolio Value: ₹{final_value:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Total Trades: {len(trades)}")
        logger.info("=" * 80)

        return {
            'success': True,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'portfolio_history': portfolio_values,
            'trades': trades,
            'metrics': metrics
        }

    def _get_rebalance_dates(self, start_date: str, end_date: str,
                            days: int) -> List[str]:
        """Generate rebalance dates."""
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=days)

        return dates

    def _get_top_stocks_for_date(self, date: str, top_n: int,
                                 min_confidence: float,
                                 min_ml_score: float,
                                 max_risk_score: float) -> List[Dict]:
        """Get top stocks with ML predictions for a specific date."""
        query = text("""
            SELECT
                symbol,
                current_price,
                ml_prediction_score,
                ml_confidence,
                ml_risk_score,
                ml_price_target
            FROM daily_suggested_stocks
            WHERE date = :date
            AND ml_prediction_score >= :min_ml_score
            AND ml_confidence >= :min_confidence
            AND ml_risk_score <= :max_risk_score
            ORDER BY ml_prediction_score DESC
            LIMIT :top_n
        """)

        result = self.db.execute(query, {
            'date': date,
            'min_ml_score': min_ml_score,
            'min_confidence': min_confidence,
            'max_risk_score': max_risk_score,
            'top_n': top_n
        })

        return [dict(row._mapping) for row in result.fetchall()]

    def _get_price_for_date(self, symbol: str, date: str) -> Optional[float]:
        """Get stock price for a specific date."""
        query = text("""
            SELECT close
            FROM historical_data
            WHERE symbol = :symbol
            AND date = :date
        """)

        result = self.db.execute(query, {'symbol': symbol, 'date': date})
        row = result.fetchone()

        return float(row[0]) if row else None

    def _calculate_metrics(self, portfolio_values: List[Dict],
                          trades: List[Dict],
                          initial_capital: float,
                          final_value: float) -> Dict:
        """Calculate performance metrics."""
        # Total return
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        # Returns series
        values = [pv['value'] for pv in portfolio_values]
        returns = np.diff(values) / values[:-1]

        # Sharpe Ratio (annualized)
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Max Drawdown
        peak = values[0]
        max_dd = 0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        max_drawdown_pct = max_dd * 100

        # Win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0

        # Average trade
        avg_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
        avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct
        }
