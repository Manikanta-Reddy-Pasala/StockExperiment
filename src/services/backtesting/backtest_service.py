"""
Backtesting Service
Simulates trading strategies on historical data to validate performance
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text

from .performance_metrics import PerformanceMetrics, calculate_performance_metrics, print_performance_report

logger = logging.getLogger(__name__)


class BacktestService:
    """
    Backtesting service for validating trading strategies.

    Simulates trading on historical data to measure:
    - Win rate
    - Average gain/loss
    - Sharpe ratio
    - Maximum drawdown
    - And other performance metrics
    """

    def __init__(self, session: Session):
        self.session = session

    def backtest_technical_strategy(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy: str = 'DEFAULT_RISK',
        hold_days: int = 5,
        initial_capital: float = 100000.0,
        position_size_pct: float = 20.0,  # % of capital per trade
        max_positions: int = 5,
        stop_loss_pct: Optional[float] = None,
        target_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Backtest the technical indicator strategy on historical data.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            strategy: Strategy to test ('DEFAULT_RISK' or 'HIGH_RISK')
            hold_days: Number of days to hold each position
            initial_capital: Starting capital
            position_size_pct: Percentage of capital per position
            max_positions: Maximum number of concurrent positions
            stop_loss_pct: Stop loss percentage (optional)
            target_pct: Target profit percentage (optional)

        Returns:
            Dict with backtest results and performance metrics
        """
        try:
            logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
            logger.info(f"Strategy: {strategy}, Hold: {hold_days} days, Capital: ₹{initial_capital:,.0f}")

            # Get all trading days in the backtest period
            trading_days = self._get_trading_days(start_date, end_date)
            logger.info(f"Found {len(trading_days)} trading days")

            if len(trading_days) < hold_days + 10:
                logger.error("Insufficient trading days for backtest")
                return {'error': 'Insufficient trading days'}

            # Initialize tracking
            all_trades = []
            capital = initial_capital
            peak_capital = initial_capital
            current_positions = []

            # Simulate trading day by day
            for i, current_date in enumerate(trading_days):
                # Skip if too close to end (need hold_days for exit)
                if i + hold_days >= len(trading_days):
                    break

                # Close expired positions
                exit_date = current_date
                positions_to_close = []

                for pos in current_positions:
                    entry_date = pos['entry_date']
                    days_held = (exit_date - entry_date).days

                    if days_held >= hold_days:
                        positions_to_close.append(pos)

                # Execute exits
                for pos in positions_to_close:
                    trade_result = self._close_position(
                        pos,
                        exit_date,
                        stop_loss_pct,
                        target_pct
                    )

                    if trade_result:
                        all_trades.append(trade_result)
                        capital += trade_result['pnl']
                        peak_capital = max(peak_capital, capital)
                        current_positions.remove(pos)

                        logger.debug(
                            f"{trade_result['exit_date'].date()}: Closed {trade_result['symbol']} "
                            f"- Return: {trade_result['return_pct']:.2f}%, Capital: ₹{capital:,.0f}"
                        )

                # Check if we can open new positions
                available_slots = max_positions - len(current_positions)
                if available_slots <= 0:
                    continue

                # Get stock picks for this date
                stock_picks = self._get_stock_picks_for_date(current_date, strategy, available_slots)

                # Open new positions
                for pick in stock_picks:
                    if len(current_positions) >= max_positions:
                        break

                    position_size = capital * (position_size_pct / 100)
                    if position_size < 1000:  # Minimum position size
                        continue

                    position = self._open_position(
                        pick['symbol'],
                        current_date,
                        pick['current_price'],
                        position_size,
                        pick
                    )

                    if position:
                        current_positions.append(position)
                        logger.debug(
                            f"{current_date.date()}: Opened {pick['symbol']} @ ₹{pick['current_price']:.2f} "
                            f"- Score: {pick.get('selection_score', 0):.1f}"
                        )

                # Progress logging
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Progress: {i+1}/{len(trading_days)} days | "
                        f"Trades: {len(all_trades)} | "
                        f"Capital: ₹{capital:,.0f} ({((capital/initial_capital - 1) * 100):.2f}%)"
                    )

            # Close any remaining positions at end
            for pos in current_positions:
                trade_result = self._close_position(pos, trading_days[-1], stop_loss_pct, target_pct)
                if trade_result:
                    all_trades.append(trade_result)
                    capital += trade_result['pnl']

            # Calculate performance metrics
            logger.info(f"Backtest complete. Total trades: {len(all_trades)}")

            if not all_trades:
                logger.warning("No trades executed during backtest period")
                return {
                    'error': 'No trades executed',
                    'trades': [],
                    'metrics': PerformanceMetrics()
                }

            metrics = calculate_performance_metrics(all_trades, initial_capital)

            # Create detailed results
            results = {
                'success': True,
                'strategy': strategy,
                'period': {
                    'start_date': start_date.date().isoformat(),
                    'end_date': end_date.date().isoformat(),
                    'trading_days': len(trading_days)
                },
                'parameters': {
                    'hold_days': hold_days,
                    'initial_capital': initial_capital,
                    'position_size_pct': position_size_pct,
                    'max_positions': max_positions,
                    'stop_loss_pct': stop_loss_pct,
                    'target_pct': target_pct
                },
                'trades': all_trades,
                'metrics': metrics,
                'final_capital': capital,
                'total_return_pct': ((capital / initial_capital) - 1) * 100
            }

            return results

        except Exception as e:
            logger.error(f"Error during backtest: {e}", exc_info=True)
            return {'error': str(e)}

    def _get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading days (days with market data) in the period."""
        try:
            query = text("""
                SELECT DISTINCT date
                FROM historical_data
                WHERE date >= :start_date
                AND date <= :end_date
                ORDER BY date ASC
            """)

            result = self.session.execute(query, {
                'start_date': start_date.date(),
                'end_date': end_date.date()
            })

            dates = [row[0] for row in result]
            return [datetime.combine(d, datetime.min.time()) for d in dates]

        except Exception as e:
            logger.error(f"Error getting trading days: {e}")
            return []

    def _get_stock_picks_for_date(
        self,
        date: datetime,
        strategy: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Simulate stock selection for a specific date using technical indicators.

        This recreates the selection logic as it would have run on that date.
        """
        try:
            # We need to calculate technical indicators using ONLY data available up to that date
            # For backtesting, we'll use a simplified approach:
            # Get stocks that would have been selected based on criteria

            query = text("""
                WITH stock_data AS (
                    SELECT
                        s.symbol,
                        s.current_price,
                        s.volume,
                        s.market_cap,
                        s.pe_ratio,
                        s.pb_ratio,
                        s.roe,
                        s.sector,
                        h.close as latest_close,
                        h.volume as latest_volume
                    FROM stocks s
                    INNER JOIN historical_data h ON s.symbol = h.symbol
                    WHERE h.date = :date
                    AND s.is_active = TRUE
                    AND s.is_tradeable = TRUE
                )
                SELECT
                    symbol,
                    latest_close as current_price,
                    latest_volume as volume,
                    market_cap,
                    pe_ratio,
                    pb_ratio,
                    roe,
                    sector,
                    -- Simple momentum score (for demo purposes)
                    latest_close * latest_volume as momentum_score
                FROM stock_data
                WHERE latest_close IS NOT NULL
                AND latest_close > 0
                AND latest_volume > 0
                ORDER BY momentum_score DESC
                LIMIT :limit
            """)

            result = self.session.execute(query, {
                'date': date.date(),
                'limit': limit
            })

            picks = []
            for row in result:
                picks.append({
                    'symbol': row[0],
                    'current_price': float(row[1]),
                    'volume': int(row[2]) if row[2] else 0,
                    'market_cap': float(row[3]) if row[3] else 0,
                    'pe_ratio': float(row[4]) if row[4] else None,
                    'pb_ratio': float(row[5]) if row[5] else None,
                    'roe': float(row[6]) if row[6] else None,
                    'sector': row[7],
                    'selection_score': 75.0  # Placeholder
                })

            return picks

        except Exception as e:
            logger.error(f"Error getting stock picks for {date.date()}: {e}")
            return []

    def _open_position(
        self,
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        position_size: float,
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Open a new position."""
        try:
            shares = int(position_size / entry_price)
            if shares <= 0:
                return None

            return {
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'shares': shares,
                'position_value': shares * entry_price,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            return None

    def _close_position(
        self,
        position: Dict[str, Any],
        exit_date: datetime,
        stop_loss_pct: Optional[float],
        target_pct: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Close a position and calculate P&L."""
        try:
            symbol = position['symbol']
            entry_price = position['entry_price']
            shares = position['shares']

            # Get exit price from historical data
            exit_price = self._get_price_on_date(symbol, exit_date)

            if exit_price is None or exit_price <= 0:
                logger.warning(f"Could not get exit price for {symbol} on {exit_date.date()}")
                return None

            # Check if stop loss or target was hit during holding period
            if stop_loss_pct or target_pct:
                actual_exit_price, actual_exit_date = self._check_stop_loss_target(
                    symbol,
                    position['entry_date'],
                    exit_date,
                    entry_price,
                    stop_loss_pct,
                    target_pct
                )
                if actual_exit_price:
                    exit_price = actual_exit_price
                    exit_date = actual_exit_date

            # Calculate return
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl = (exit_price - entry_price) * shares

            # Hold days
            hold_days = (exit_date - position['entry_date']).days

            return {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'return_pct': return_pct,
                'pnl': pnl,
                'hold_days': hold_days,
                'metadata': position['metadata']
            }

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def _get_price_on_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get stock price on a specific date."""
        try:
            query = text("""
                SELECT close
                FROM historical_data
                WHERE symbol = :symbol
                AND date = :date
                LIMIT 1
            """)

            result = self.session.execute(query, {
                'symbol': symbol,
                'date': date.date()
            }).fetchone()

            return float(result[0]) if result else None

        except Exception as e:
            logger.error(f"Error getting price for {symbol} on {date.date()}: {e}")
            return None

    def _check_stop_loss_target(
        self,
        symbol: str,
        entry_date: datetime,
        exit_date: datetime,
        entry_price: float,
        stop_loss_pct: Optional[float],
        target_pct: Optional[float]
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """
        Check if stop loss or target was hit during holding period.

        Returns:
            Tuple of (actual_exit_price, actual_exit_date) if hit, else (None, None)
        """
        if not stop_loss_pct and not target_pct:
            return None, None

        try:
            # Get daily prices during holding period
            query = text("""
                SELECT date, high, low
                FROM historical_data
                WHERE symbol = :symbol
                AND date > :entry_date
                AND date <= :exit_date
                ORDER BY date ASC
            """)

            result = self.session.execute(query, {
                'symbol': symbol,
                'entry_date': entry_date.date(),
                'exit_date': exit_date.date()
            })

            for row in result:
                date, high, low = row

                # Check target hit
                if target_pct and high >= entry_price * (1 + target_pct / 100):
                    return entry_price * (1 + target_pct / 100), datetime.combine(date, datetime.min.time())

                # Check stop loss hit
                if stop_loss_pct and low <= entry_price * (1 - stop_loss_pct / 100):
                    return entry_price * (1 - stop_loss_pct / 100), datetime.combine(date, datetime.min.time())

            return None, None

        except Exception as e:
            logger.error(f"Error checking stop loss/target: {e}")
            return None, None

    def run_backtest_and_print_report(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> None:
        """Run backtest and print formatted report."""
        results = self.backtest_technical_strategy(start_date, end_date, **kwargs)

        if 'error' in results:
            print(f"\n❌ Backtest Error: {results['error']}\n")
            return

        # Print summary
        print("\n" + "=" * 80)
        print("BACKTEST CONFIGURATION")
        print("=" * 80)
        print(f"  Strategy:         {results['strategy']}")
        print(f"  Period:           {results['period']['start_date']} to {results['period']['end_date']}")
        print(f"  Trading Days:     {results['period']['trading_days']}")
        print(f"  Hold Days:        {results['parameters']['hold_days']}")
        print(f"  Initial Capital:  ₹{results['parameters']['initial_capital']:,.0f}")
        print(f"  Position Size:    {results['parameters']['position_size_pct']:.1f}%")
        print(f"  Max Positions:    {results['parameters']['max_positions']}")
        print("=" * 80)

        # Print performance report
        print_performance_report(results['metrics'])

        # Print sample trades
        print("\n📋 SAMPLE TRADES (First 10)")
        print("-" * 80)
        for i, trade in enumerate(results['trades'][:10]):
            status = "✅" if trade['return_pct'] > 0 else "❌"
            print(
                f"{i+1:3d}. {status} {trade['symbol']:20s} | "
                f"Entry: {trade['entry_date'].date()} @ ₹{trade['entry_price']:7.2f} | "
                f"Exit: {trade['exit_date'].date()} @ ₹{trade['exit_price']:7.2f} | "
                f"Return: {trade['return_pct']:6.2f}% | "
                f"Hold: {trade['hold_days']} days"
            )
        if len(results['trades']) > 10:
            print(f"  ... and {len(results['trades']) - 10} more trades")
        print("=" * 80 + "\n")


# Singleton instance
_backtest_service = None


def get_backtest_service() -> BacktestService:
    """Get backtest service instance (requires session to be provided)."""
    # This is a factory function that requires session
    # Actual instantiation happens in calling code
    pass


def create_backtest_service(session: Session) -> BacktestService:
    """Create a new backtest service instance."""
    return BacktestService(session)
