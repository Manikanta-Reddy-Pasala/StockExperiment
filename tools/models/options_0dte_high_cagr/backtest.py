#!/usr/bin/env python3
"""
Multi-Index 0DTE Iron-Fly Backtest Simulation

This script simulates the logic for the `options_0dte_high_cagr` model.
Due to the absence of full intraday options data in the historical database,
this code acts as a conceptual backtest and scaffolding for the strategy rules.

It demonstrates how the capital compounds over multiple expiries.
"""

import argparse
import random
import json
from datetime import date, timedelta
from typing import List, Dict

# Model parameters
WIN_RATE = 0.72  # Assumed historical win rate for 0DTE Iron Fly
AVG_WIN_PCT = 0.04  # 4% return on deployed margin
AVG_LOSS_PCT = -0.06 # -6% return on deployed margin (capped by wings)
TRADES_PER_WEEK = 3  # Nifty, BankNifty, Finnifty

def simulate_trade(capital_deployed: float) -> float:
    """Simulates a single 0DTE Iron Fly outcome."""
    is_win = random.random() <= WIN_RATE
    if is_win:
        return capital_deployed * AVG_WIN_PCT
    else:
        return capital_deployed * AVG_LOSS_PCT

def run_backtest(start_date: date, end_date: date, initial_capital: float) -> Dict:
    """
    Simulates the compounded returns of the 0DTE multi-index strategy.
    In a real environment, this would query the `historical_options` table.
    """
    current_date = start_date
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0

    trades = []
    wins = 0
    losses = 0

    while current_date <= end_date:
        # Trade on Tuesday (Finnifty), Wednesday (BankNifty), Thursday (Nifty)
        if current_date.weekday() in [1, 2, 3]:
            # Deploy 80% of available capital as margin (leaving room for minor fluctuations)
            deployed_margin = capital * 0.80

            pnl = simulate_trade(deployed_margin)
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
                "deployed_margin": round(deployed_margin, 2),
                "pnl": round(pnl, 2),
                "capital_after": round(capital, 2)
            })

        current_date += timedelta(days=1)

    years = (end_date - start_date).days / 365.25
    cagr = ((capital / initial_capital) ** (1 / max(years, 1e-9)) - 1) * 100

    return {
        "model": "options_0dte_high_cagr",
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
    parser = argparse.ArgumentParser(description="Run 0DTE Multi-Index Backtest")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--days", type=int, default=365, help="Number of days to simulate")

    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    # Run the simulation
    # Using a fixed seed for reproducible simulation results if needed,
    # but left random here to show statistical expectancy.
    results = run_backtest(start, end, args.capital)

    print(json.dumps(results, indent=2))
