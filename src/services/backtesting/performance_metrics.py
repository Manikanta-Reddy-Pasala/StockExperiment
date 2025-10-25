"""
Performance Metrics Calculator
Calculates comprehensive performance metrics for backtesting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results."""

    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Return metrics
    total_return: float = 0.0
    avg_return_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0

    # Advanced metrics
    profit_factor: float = 0.0  # Total wins / Total losses
    expectancy: float = 0.0     # (Win% × Avg Win) - (Loss% × Avg Loss)
    calmar_ratio: float = 0.0   # Total Return / Max Drawdown

    # Time-based metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: int = 0

    # Portfolio metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0

    # Trade duration metrics
    avg_hold_days: float = 0.0
    min_hold_days: int = 0
    max_hold_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'basic_metrics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': f"{self.win_rate:.2f}%"
            },
            'return_metrics': {
                'total_return': f"{self.total_return:.2f}%",
                'avg_return_per_trade': f"{self.avg_return_per_trade:.2f}%",
                'avg_win': f"{self.avg_win:.2f}%",
                'avg_loss': f"{self.avg_loss:.2f}%",
                'largest_win': f"{self.largest_win:.2f}%",
                'largest_loss': f"{self.largest_loss:.2f}%"
            },
            'risk_metrics': {
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'sortino_ratio': f"{self.sortino_ratio:.2f}",
                'max_drawdown': f"{self.max_drawdown:.2f}%",
                'max_drawdown_duration': f"{self.max_drawdown_duration} days",
                'volatility': f"{self.volatility:.2f}%"
            },
            'advanced_metrics': {
                'profit_factor': f"{self.profit_factor:.2f}",
                'expectancy': f"{self.expectancy:.2f}%",
                'calmar_ratio': f"{self.calmar_ratio:.2f}"
            },
            'portfolio_metrics': {
                'initial_capital': f"₹{self.initial_capital:,.0f}",
                'final_capital': f"₹{self.final_capital:,.0f}",
                'peak_capital': f"₹{self.peak_capital:,.0f}"
            },
            'duration_metrics': {
                'avg_hold_days': f"{self.avg_hold_days:.1f}",
                'min_hold_days': self.min_hold_days,
                'max_hold_days': self.max_hold_days,
                'total_days': self.total_days
            }
        }


def calculate_performance_metrics(
    trades: List[Dict[str, Any]],
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.065  # 6.5% annual risk-free rate (FD rate in India)
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from trade history.

    Args:
        trades: List of trade dictionaries with keys:
            - symbol: Stock symbol
            - entry_date: Entry date
            - exit_date: Exit date
            - entry_price: Entry price
            - exit_price: Exit price
            - return_pct: Return percentage
            - hold_days: Days held
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino

    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    if not trades:
        logger.warning("No trades provided for performance calculation")
        return PerformanceMetrics()

    try:
        metrics = PerformanceMetrics()

        # Extract returns
        returns = [t['return_pct'] for t in trades]
        hold_days_list = [t.get('hold_days', 0) for t in trades]

        # Basic metrics
        metrics.total_trades = len(trades)
        metrics.winning_trades = sum(1 for r in returns if r > 0)
        metrics.losing_trades = sum(1 for r in returns if r < 0)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades * 100) if metrics.total_trades > 0 else 0

        # Return metrics
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        metrics.total_return = sum(returns)
        metrics.avg_return_per_trade = np.mean(returns) if returns else 0
        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.largest_win = max(returns) if returns else 0
        metrics.largest_loss = min(returns) if returns else 0

        # Calculate cumulative returns for drawdown
        cumulative_returns = np.cumsum(returns)

        # Risk metrics
        if len(returns) > 1:
            # Sharpe Ratio (annualized)
            avg_daily_return = np.mean(returns)
            std_daily_return = np.std(returns, ddof=1)

            # Convert to annual metrics (assuming ~252 trading days)
            avg_annual_return = avg_daily_return * 252 / np.mean(hold_days_list) if hold_days_list else 0
            annual_std = std_daily_return * np.sqrt(252 / np.mean(hold_days_list)) if hold_days_list else 0

            if annual_std > 0:
                metrics.sharpe_ratio = (avg_annual_return - risk_free_rate) / annual_std

            # Sortino Ratio (only downside volatility)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns, ddof=1)
                annual_downside_std = downside_std * np.sqrt(252 / np.mean(hold_days_list)) if hold_days_list else 0
                if annual_downside_std > 0:
                    metrics.sortino_ratio = (avg_annual_return - risk_free_rate) / annual_downside_std

            # Volatility (annualized)
            metrics.volatility = annual_std * 100  # Convert to percentage

        # Maximum Drawdown
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Max drawdown duration
            in_drawdown = drawdown > 0
            if np.any(in_drawdown):
                drawdown_periods = np.split(np.arange(len(in_drawdown)), np.where(~in_drawdown)[0])
                drawdown_periods = [p for p in drawdown_periods if len(p) > 0 and np.all(in_drawdown[p])]
                if drawdown_periods:
                    max_dd_period = max(drawdown_periods, key=len)
                    metrics.max_drawdown_duration = sum(hold_days_list[i] for i in max_dd_period if i < len(hold_days_list))

        # Advanced metrics
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')

        # Expectancy
        win_prob = metrics.win_rate / 100
        loss_prob = 1 - win_prob
        metrics.expectancy = (win_prob * metrics.avg_win) - (loss_prob * abs(metrics.avg_loss))

        # Calmar Ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.total_return / metrics.max_drawdown

        # Time-based metrics
        if trades:
            dates = [t['entry_date'] for t in trades if 'entry_date' in t]
            if dates:
                metrics.start_date = min(dates)
                metrics.end_date = max(dates)
                metrics.total_days = (metrics.end_date - metrics.start_date).days if metrics.end_date and metrics.start_date else 0

        # Portfolio metrics
        metrics.initial_capital = initial_capital
        metrics.final_capital = initial_capital * (1 + metrics.total_return / 100)

        # Calculate peak capital
        capital_curve = [initial_capital]
        for ret in returns:
            capital_curve.append(capital_curve[-1] * (1 + ret / 100))
        metrics.peak_capital = max(capital_curve)

        # Trade duration metrics
        if hold_days_list:
            metrics.avg_hold_days = np.mean(hold_days_list)
            metrics.min_hold_days = min(hold_days_list)
            metrics.max_hold_days = max(hold_days_list)

        return metrics

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
        return PerformanceMetrics()


def print_performance_report(metrics: PerformanceMetrics) -> None:
    """Print a formatted performance report."""
    print("\n" + "=" * 80)
    print("BACKTEST PERFORMANCE REPORT")
    print("=" * 80)

    print("\n📊 BASIC METRICS")
    print("-" * 80)
    print(f"  Total Trades:       {metrics.total_trades}")
    print(f"  Winning Trades:     {metrics.winning_trades}")
    print(f"  Losing Trades:      {metrics.losing_trades}")
    print(f"  Win Rate:           {metrics.win_rate:.2f}%")

    print("\n💰 RETURN METRICS")
    print("-" * 80)
    print(f"  Total Return:       {metrics.total_return:.2f}%")
    print(f"  Avg Return/Trade:   {metrics.avg_return_per_trade:.2f}%")
    print(f"  Average Win:        {metrics.avg_win:.2f}%")
    print(f"  Average Loss:       {metrics.avg_loss:.2f}%")
    print(f"  Largest Win:        {metrics.largest_win:.2f}%")
    print(f"  Largest Loss:       {metrics.largest_loss:.2f}%")

    print("\n⚠️  RISK METRICS")
    print("-" * 80)
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:.2f}%")
    print(f"  DD Duration:        {metrics.max_drawdown_duration} days")
    print(f"  Volatility (Ann.):  {metrics.volatility:.2f}%")

    print("\n📈 ADVANCED METRICS")
    print("-" * 80)
    print(f"  Profit Factor:      {metrics.profit_factor:.2f}")
    print(f"  Expectancy:         {metrics.expectancy:.2f}%")
    print(f"  Calmar Ratio:       {metrics.calmar_ratio:.2f}")

    print("\n💼 PORTFOLIO METRICS")
    print("-" * 80)
    print(f"  Initial Capital:    ₹{metrics.initial_capital:,.0f}")
    print(f"  Final Capital:      ₹{metrics.final_capital:,.0f}")
    print(f"  Peak Capital:       ₹{metrics.peak_capital:,.0f}")
    print(f"  P&L:                ₹{metrics.final_capital - metrics.initial_capital:,.0f}")

    print("\n⏱️  DURATION METRICS")
    print("-" * 80)
    print(f"  Avg Hold Days:      {metrics.avg_hold_days:.1f}")
    print(f"  Min Hold Days:      {metrics.min_hold_days}")
    print(f"  Max Hold Days:      {metrics.max_hold_days}")
    print(f"  Test Period:        {metrics.total_days} days")
    if metrics.start_date and metrics.end_date:
        print(f"  Start Date:         {metrics.start_date.strftime('%Y-%m-%d')}")
        print(f"  End Date:           {metrics.end_date.strftime('%Y-%m-%d')}")

    print("\n" + "=" * 80)

    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("-" * 80)

    assessments = []

    if metrics.win_rate >= 60:
        assessments.append("✅ Excellent win rate (>60%)")
    elif metrics.win_rate >= 50:
        assessments.append("✅ Good win rate (50-60%)")
    else:
        assessments.append("⚠️  Low win rate (<50%)")

    if metrics.sharpe_ratio >= 1.5:
        assessments.append("✅ Excellent risk-adjusted returns (Sharpe >1.5)")
    elif metrics.sharpe_ratio >= 1.0:
        assessments.append("✅ Good risk-adjusted returns (Sharpe 1.0-1.5)")
    elif metrics.sharpe_ratio >= 0.5:
        assessments.append("⚠️  Moderate risk-adjusted returns (Sharpe 0.5-1.0)")
    else:
        assessments.append("❌ Poor risk-adjusted returns (Sharpe <0.5)")

    if metrics.max_drawdown < 10:
        assessments.append("✅ Low drawdown risk (<10%)")
    elif metrics.max_drawdown < 20:
        assessments.append("⚠️  Moderate drawdown risk (10-20%)")
    else:
        assessments.append("❌ High drawdown risk (>20%)")

    if metrics.profit_factor >= 2.0:
        assessments.append("✅ Excellent profit factor (>2.0)")
    elif metrics.profit_factor >= 1.5:
        assessments.append("✅ Good profit factor (1.5-2.0)")
    elif metrics.profit_factor >= 1.0:
        assessments.append("⚠️  Moderate profit factor (1.0-1.5)")
    else:
        assessments.append("❌ Poor profit factor (<1.0)")

    for assessment in assessments:
        print(f"  {assessment}")

    print("=" * 80 + "\n")
