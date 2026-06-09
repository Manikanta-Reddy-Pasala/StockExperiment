#!/usr/bin/env python3
"""
Index Options Momentum Breakout Backtest Simulation

This script simulates the logic for the `index_options_momentum` model.
It models the asymmetric reward-to-risk profile of buying ATM options
on momentum bursts, strictly limiting position sizing to control drawdowns.
"""

import argparse
import random
import json
from datetime import date, timedelta
from typing import List, Dict

# Model parameters
WIN_RATE = 0.42 # 42% win rate on breakouts
TARGET_PREMIUM_GAIN = 0.50 # 50% gain on premium
STOP_PREMIUM_LOSS = -0.30 # 30% loss on premium
CAPITAL_ALLOCATION = 0.10 # Max 10% of total portfolio deployed into the option per trade
TRADES_PER_WEEK = 3 # Average number of qualifying breakouts per week

def simulate_trade(capital_allocated: float) -> float:
    """Simulates a single directional option buying trade outcome."""
    is_win = random.random() <= WIN_RATE
    if is_win:
        return capital_allocated * TARGET_PREMIUM_GAIN
    else:
        return capital_allocated * STOP_PREMIUM_LOSS

def run_backtest(start_date: date, end_date: date, initial_capital: float) -> Dict:
    """
    Simulates the compounded returns of the Directional Options strategy.
    """
    current_date = start_date
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0

    trades = []
    wins = 0
    losses = 0

    while current_date <= end_date:
        # Simulate trading roughly 3 times a week (probabilistic mapping)
        if current_date.weekday() < 5 and random.random() < (TRADES_PER_WEEK / 5.0):
            # Allocate strictly 10% of available capital
            deployed_capital = capital * CAPITAL_ALLOCATION

            pnl = simulate_trade(deployed_capital)
            capital += pnl

            if pnl > 0:
                wins += 1
            else:
                losses += 1

            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            trades.append({
                "date": current_date.isoformat(),
                "deployed_capital": round(deployed_capital, 2),
                "pnl": round(pnl, 2),
                "capital_after": round(capital, 2)
            })

        current_date += timedelta(days=1)

    years = (end_date - start_date).days / 365.25
    cagr = ((capital / initial_capital) ** (1 / max(years, 1e-9)) - 1) * 100

    return {
        "model": "index_options_momentum",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "total_trades": len(trades),
        "win_rate_pct": round(wins / max(len(trades), 1) * 100, 2),
        "trades_sample": trades[-5:] # Show last 5 trades
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Index Options Momentum Backtest")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--days", type=int, default=365, help="Number of days to simulate")

    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    # Run the simulation
    results = run_backtest(start, end, args.capital)

    print(json.dumps(results, indent=2))
