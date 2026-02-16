#!/usr/bin/env python3
"""
Comprehensive Backtest & Fine-Tuning for 8-21 EMA Recommendation Engine

Uses Oct 2025 data for signal generation and Nov 2025 data for forward return measurement.
Tests multiple parameter combinations to find optimal thresholds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from itertools import product

# ===== CONSTANTS =====
ETF_EXCLUSION_SYMBOLS = {
    'LIQUIDBEES', 'NIFTYBEES', 'BANKBEES', 'MON100', 'GOLDBEES',
    'SILVERBEES', 'JUNIORBEES', 'ITBEES', 'PSUBNKBEES', 'SETFNIF50',
    'SETFNIFBK', 'NETFIT', 'HABOREES', 'HNGSNGBEES', 'SHARIABEES',
    'DIVOPPBEES', 'INFRABEES', 'CPSEETF', 'CONSUMBEES', 'HEALTHY',
    'MOM100', 'MOM50', 'PHARMABEES', 'AUTOBEES', 'COMMOETF',
}


def clean_symbol(raw):
    return raw.upper().replace('NSE:', '').replace('BSE:', '').replace('-EQ', '').strip()


def is_etf(symbol, name):
    clean = clean_symbol(symbol)
    n = (name or '').upper()
    return 'ETF' in n or 'ETF' in clean or 'BEES' in clean or clean in ETF_EXCLUSION_SYMBOLS


def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calculate_demarker(df, period=14):
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    demax = high_diff.where(high_diff > 0, 0)
    demin = (-low_diff).where(low_diff < 0, 0)
    demax_sma = demax.rolling(window=period).mean()
    demin_sma = demin.rolling(window=period).mean()
    demarker = demax_sma / (demax_sma + demin_sma)
    val = demarker.iloc[-1]
    return max(0.0, min(1.0, val)) if not pd.isna(val) else 0.5


def compute_score(price, ema_8, ema_21, demarker, weights):
    """Compute score with configurable weights."""
    if ema_21 <= 0 or price <= 0:
        return 0.0
    if not (price > ema_8 > ema_21):
        return 0.0

    w_sep, w_dm, w_dist = weights

    # EMA separation (0 to w_sep)
    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    # DeMarker timing (0 to w_dm)
    if demarker < 0.20:
        dm_score = w_dm * 1.0
    elif demarker < 0.30:
        dm_score = w_dm * 0.83
    elif demarker < 0.40:
        dm_score = w_dm * 0.60
    elif demarker < 0.50:
        dm_score = w_dm * 0.40
    elif demarker < 0.70:
        dm_score = w_dm * 0.17
    else:
        dm_score = 0.0

    # Price distance from EMA8 (0 to w_dist)
    price_dist_pct = ((price - ema_8) / ema_8) * 100
    if 0 <= price_dist_pct <= 1.0:
        dist_score = w_dist * 1.0
    elif price_dist_pct <= 2.0:
        dist_score = w_dist * 0.83
    elif price_dist_pct <= 3.0:
        dist_score = w_dist * 0.60
    elif price_dist_pct <= 5.0:
        dist_score = w_dist * 0.33
    else:
        dist_score = w_dist * 0.10

    return round(sep_score + dm_score + dist_score, 2)


def get_signal_quality(demarker, quality_thresholds):
    """Get signal quality based on DeMarker and configurable thresholds."""
    high_th, med_th = quality_thresholds
    if demarker < high_th:
        return 'high'
    elif demarker < med_th:
        return 'medium'
    return 'low'


def run_backtest(entry_indicators, forward_returns, params):
    """
    Run a single backtest with given parameters.

    params = {
        'weights': (sep_w, dm_w, dist_w),  # must sum to 100
        'min_score': float,
        'quality_thresholds': (high_th, med_th),
        'accepted_qualities': set,
        'min_volume': int,
        'min_market_cap': float,
        'top_n': int,
    }
    """
    weights = params['weights']
    min_score = params['min_score']
    quality_thresholds = params['quality_thresholds']
    accepted_qualities = params['accepted_qualities']
    top_n = params['top_n']

    # Score and filter
    candidates = []
    for _, row in entry_indicators.iterrows():
        if row['is_etf']:
            continue
        if row['price'] < 50 or row['price'] > 10000:
            continue
        if row['volume'] < params.get('min_volume', 50000):
            continue
        if row['market_cap'] < params.get('min_market_cap', 500):
            continue

        # Power zone check
        if not (row['price'] > row['ema_8'] > row['ema_21']):
            continue

        score = compute_score(row['price'], row['ema_8'], row['ema_21'],
                             row['demarker'], weights)
        quality = get_signal_quality(row['demarker'], quality_thresholds)

        if score < min_score:
            continue
        if quality not in accepted_qualities:
            continue

        candidates.append({
            'symbol': row['symbol'],
            'clean_symbol': row['clean_symbol'],
            'score': score,
            'quality': quality,
            'demarker': row['demarker'],
            'ema_sep': row['ema_sep_pct'],
            'price_dist': row['price_dist_pct'],
        })

    if not candidates:
        return None

    # Sort by score, take top N
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = candidates[:top_n]

    # Match with forward returns
    results = []
    for c in selected:
        sym = c['clean_symbol']
        if sym in forward_returns:
            fwd = forward_returns[sym]
            results.append({**c, 'forward_return': fwd})

    if not results:
        return None

    # Calculate metrics
    returns = [r['forward_return'] for r in results]
    winners = sum(1 for r in returns if r > 0)
    losers = sum(1 for r in returns if r <= 0)
    win_rate = winners / len(returns) * 100
    avg_return = np.mean(returns)
    median_return = np.median(returns)
    total_return = np.sum(returns)

    # Risk-adjusted: Sharpe-like ratio
    if np.std(returns) > 0:
        sharpe = avg_return / np.std(returns)
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'n_candidates': len(candidates),
        'n_selected': len(selected),
        'n_with_data': len(results),
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
        'median_return': round(median_return, 2),
        'total_return': round(total_return, 2),
        'sharpe': round(sharpe, 3),
        'profit_factor': round(profit_factor, 2),
        'max_gain': round(max(returns), 2),
        'max_loss': round(min(returns), 2),
        'results': results,
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE BACKTEST & FINE-TUNING")
    print("Strategy: 8-21 EMA Swing Trading")
    print("Entry signals: Oct 2025 data | Forward returns: Nov 2025 data")
    print("=" * 80)

    # ===== LOAD DATA =====
    stocks_df = pd.read_csv('/opt/clawdbot/repos/StockExperiment/exports/stocks_2025-10-10.csv')
    oct_hist = pd.read_csv('/opt/clawdbot/repos/StockExperiment/exports/historical_30d_2025-10-10.csv')
    nov_hist = pd.read_csv('/opt/clawdbot/repos/StockExperiment/exports/historical_30d_2025-11-15.csv')

    print(f"\nStocks: {len(stocks_df)}")
    print(f"Oct hist: {oct_hist['symbol'].nunique()} symbols ({oct_hist['date'].min()} to {oct_hist['date'].max()})")
    print(f"Nov hist: {nov_hist['symbol'].nunique()} symbols ({nov_hist['date'].min()} to {nov_hist['date'].max()})")

    # ===== CALCULATE ENTRY INDICATORS (from Oct data, last day) =====
    print("\nCalculating entry indicators from Oct data...")
    entry_indicators = []
    for symbol in oct_hist['symbol'].unique():
        sym_data = oct_hist[oct_hist['symbol'] == symbol].sort_values('date').copy()
        if len(sym_data) < 14:
            continue

        sym_data['ema_8'] = calculate_ema(sym_data['close'], 8)
        sym_data['ema_21'] = calculate_ema(sym_data['close'], 21)
        demarker = calculate_demarker(sym_data)

        latest = sym_data.iloc[-1]
        price = latest['close']
        ema_8 = latest['ema_8']
        ema_21 = latest['ema_21']

        stock_info = stocks_df[stocks_df['symbol'] == symbol]
        name = stock_info['name'].iloc[0] if len(stock_info) > 0 else ''
        market_cap = stock_info['market_cap'].iloc[0] if len(stock_info) > 0 and pd.notna(stock_info['market_cap'].iloc[0]) else 0
        volume = latest['volume']
        clean_sym = clean_symbol(symbol)

        ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100 if ema_21 > 0 else 0
        price_dist_pct = ((price - ema_8) / ema_8) * 100 if ema_8 > 0 else 0

        entry_indicators.append({
            'symbol': symbol,
            'clean_symbol': clean_sym,
            'name': name,
            'price': price,
            'ema_8': ema_8,
            'ema_21': ema_21,
            'demarker': demarker,
            'volume': volume,
            'market_cap': market_cap,
            'ema_sep_pct': ema_sep_pct,
            'price_dist_pct': price_dist_pct,
            'is_etf': is_etf(symbol, name),
        })

    entry_df = pd.DataFrame(entry_indicators)
    bullish = entry_df[(entry_df['price'] > entry_df['ema_8']) & (entry_df['ema_8'] > entry_df['ema_21'])]
    print(f"Calculated indicators for {len(entry_df)} stocks ({len(bullish)} in power zone)")

    # ===== CALCULATE FORWARD RETURNS (Nov data) =====
    # Forward return = first available Nov price vs last available Nov price
    # Simulates ~20 trading day hold (swing trade)
    print("Calculating forward returns from Nov data...")
    forward_returns = {}
    for symbol in nov_hist['symbol'].unique():
        sym_data = nov_hist[nov_hist['symbol'] == symbol].sort_values('date')
        if len(sym_data) < 5:
            continue
        entry_price = sym_data.iloc[0]['close']  # First Nov day = entry after Oct signal
        # Test multiple holding periods
        for hold_days, label in [(5, '5d'), (10, '10d'), (15, '15d')]:
            exit_idx = min(hold_days, len(sym_data) - 1)
            exit_price = sym_data.iloc[exit_idx]['close']
            fwd_ret = ((exit_price - entry_price) / entry_price) * 100
            clean_sym = clean_symbol(symbol)
            if clean_sym not in forward_returns:
                forward_returns[clean_sym] = {}
            forward_returns[clean_sym][label] = fwd_ret

    # Also compute max exit (best within the 20 days) and stop-loss hit
    for symbol in nov_hist['symbol'].unique():
        sym_data = nov_hist[nov_hist['symbol'] == symbol].sort_values('date')
        if len(sym_data) < 5:
            continue
        entry_price = sym_data.iloc[0]['close']
        max_price = sym_data['high'].max()
        min_price = sym_data['low'].min()
        clean_sym = clean_symbol(symbol)
        if clean_sym in forward_returns:
            forward_returns[clean_sym]['max_gain'] = ((max_price - entry_price) / entry_price) * 100
            forward_returns[clean_sym]['max_loss'] = ((min_price - entry_price) / entry_price) * 100

    print(f"Forward returns calculated for {len(forward_returns)} symbols")

    # Use 10-day hold as primary metric
    fwd_10d = {sym: data['10d'] for sym, data in forward_returns.items() if '10d' in data}

    # ===== PARAMETER GRID SEARCH =====
    print(f"\n{'='*80}")
    print("PARAMETER GRID SEARCH")
    print(f"{'='*80}")

    # Define parameter grid
    weight_combos = [
        (40, 30, 30),  # Current: balanced
        (50, 25, 25),  # More EMA separation focus
        (30, 40, 30),  # More DeMarker focus
        (30, 30, 40),  # More price distance focus
        (20, 50, 30),  # Heavy DeMarker
        (50, 30, 20),  # Heavy separation, less distance
        (35, 35, 30),  # Near-equal
        (40, 40, 20),  # Sep + DeMarker focus
    ]

    score_thresholds = [20, 25, 30, 35, 40]
    quality_configs = [
        ((0.30, 0.50), {'high', 'medium'}),       # Current
        ((0.25, 0.45), {'high', 'medium'}),        # Stricter
        ((0.30, 0.50), {'high'}),                   # High only
        ((0.35, 0.55), {'high', 'medium'}),         # Relaxed thresholds
        ((0.30, 0.70), {'high', 'medium'}),         # Wide medium band
    ]

    top_n_options = [10, 15, 20, 30, 50]

    best_results = []

    total_combos = len(weight_combos) * len(score_thresholds) * len(quality_configs) * len(top_n_options)
    print(f"Testing {total_combos} parameter combinations...")

    tested = 0
    for weights in weight_combos:
        for min_score in score_thresholds:
            for (quality_th, accepted_q) in quality_configs:
                for top_n in top_n_options:
                    params = {
                        'weights': weights,
                        'min_score': min_score,
                        'quality_thresholds': quality_th,
                        'accepted_qualities': accepted_q,
                        'min_volume': 50000,
                        'min_market_cap': 500,
                        'top_n': top_n,
                    }

                    result = run_backtest(entry_df, fwd_10d, params)
                    tested += 1

                    if result and result['n_with_data'] >= 3:
                        best_results.append({
                            'weights': weights,
                            'min_score': min_score,
                            'quality_th': quality_th,
                            'accepted_q': tuple(sorted(accepted_q)),
                            'top_n': top_n,
                            **result,
                        })

    print(f"Tested {tested} combinations, {len(best_results)} had sufficient data")

    if not best_results:
        print("ERROR: No parameter combinations produced results with enough data.")
        return

    results_df = pd.DataFrame(best_results)

    # ===== RANK BY MULTIPLE CRITERIA =====
    # Primary: win rate (must be >= 50%)
    # Secondary: profit factor
    # Tertiary: average return
    viable = results_df[results_df['win_rate'] >= 50].copy()
    if len(viable) == 0:
        print("WARNING: No combinations achieved >= 50% win rate. Showing best available.")
        viable = results_df.copy()

    # Composite ranking score
    viable['rank_score'] = (
        viable['win_rate'] * 0.3 +
        viable['avg_return'].clip(-10, 20) * 3.0 +
        viable['profit_factor'].clip(0, 5) * 5.0 +
        viable['sharpe'].clip(-2, 3) * 10.0
    )

    viable = viable.sort_values('rank_score', ascending=False)

    # ===== SHOW TOP 20 RESULTS =====
    print(f"\n{'='*80}")
    print(f"TOP 20 PARAMETER COMBINATIONS (sorted by composite rank)")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Weights':>15} {'MinScore':>8} {'QualTh':>12} {'Qual':>8} {'TopN':>5} | "
          f"{'WinR%':>6} {'AvgR%':>6} {'MedR%':>6} {'PF':>5} {'Shrp':>5} | {'N':>3} {'MaxG%':>6} {'MaxL%':>6}")
    print("-" * 120)

    for i, (_, row) in enumerate(viable.head(20).iterrows(), 1):
        print(f"{i:3d} {str(row['weights']):>15} {row['min_score']:>8} {str(row['quality_th']):>12} "
              f"{str(row['accepted_q']):>8} {row['top_n']:>5} | "
              f"{row['win_rate']:>6.1f} {row['avg_return']:>6.2f} {row['median_return']:>6.2f} "
              f"{row['profit_factor']:>5.2f} {row['sharpe']:>5.3f} | "
              f"{row['n_with_data']:>3} {row['max_gain']:>6.2f} {row['max_loss']:>6.2f}")

    # ===== BEST CONFIGURATION =====
    best = viable.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Weights:              {best['weights']}  (EMA separation, DeMarker, Price distance)")
    print(f"  Min score threshold:  {best['min_score']}")
    print(f"  Quality thresholds:   DeMarker < {best['quality_th'][0]} = high, < {best['quality_th'][1]} = medium")
    print(f"  Accepted qualities:   {best['accepted_q']}")
    print(f"  Top N selections:     {best['top_n']}")
    print(f"\n  PERFORMANCE:")
    print(f"  Win rate:         {best['win_rate']:.1f}%")
    print(f"  Avg return:       {best['avg_return']:+.2f}%")
    print(f"  Median return:    {best['median_return']:+.2f}%")
    print(f"  Profit factor:    {best['profit_factor']:.2f}")
    print(f"  Sharpe ratio:     {best['sharpe']:.3f}")
    print(f"  Stocks tested:    {best['n_with_data']}")
    print(f"  Max gain:         {best['max_gain']:+.2f}%")
    print(f"  Max loss:         {best['max_loss']:+.2f}%")

    # ===== SHOW INDIVIDUAL STOCKS FOR BEST CONFIG =====
    best_params = {
        'weights': best['weights'],
        'min_score': best['min_score'],
        'quality_thresholds': best['quality_th'],
        'accepted_qualities': set(best['accepted_q']),
        'min_volume': 50000,
        'min_market_cap': 500,
        'top_n': best['top_n'],
    }
    best_bt = run_backtest(entry_df, fwd_10d, best_params)

    print(f"\n{'='*80}")
    print(f"BEST CONFIG - INDIVIDUAL STOCK RESULTS")
    print(f"{'='*80}")
    print(f"{'Symbol':15s} {'Score':>6} {'Qual':>6} {'DeM':>5} {'EMASep%':>8} {'PriceDist%':>10} {'Return%':>8} {'Result':>8}")
    print("-" * 80)

    for r in sorted(best_bt['results'], key=lambda x: x['forward_return'], reverse=True):
        result_str = "WIN" if r['forward_return'] > 0 else "LOSS"
        print(f"{r['clean_symbol']:15s} {r['score']:>6.1f} {r['quality']:>6} {r['demarker']:>5.2f} "
              f"{r['ema_sep']:>8.2f} {r['price_dist']:>10.2f} {r['forward_return']:>+8.2f} {result_str:>8}")

    # ===== MULTI-HOLDING PERIOD ANALYSIS =====
    print(f"\n{'='*80}")
    print(f"HOLDING PERIOD ANALYSIS (Best Config)")
    print(f"{'='*80}")

    for hold_label in ['5d', '10d', '15d']:
        fwd_data = {sym: data[hold_label] for sym, data in forward_returns.items() if hold_label in data}
        hold_bt = run_backtest(entry_df, fwd_data, best_params)
        if hold_bt:
            print(f"  {hold_label:>4s} hold: WinRate={hold_bt['win_rate']:>5.1f}%  "
                  f"AvgReturn={hold_bt['avg_return']:>+6.2f}%  "
                  f"PF={hold_bt['profit_factor']:>5.2f}  "
                  f"Stocks={hold_bt['n_with_data']}")

    # ===== FEATURE IMPORTANCE: What predicts winners? =====
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE: What predicts winners?")
    print(f"{'='*80}")

    # Get all power zone stocks with forward returns
    analysis_data = []
    for _, row in entry_df.iterrows():
        if row['is_etf'] or row['price'] < 50 or row['price'] > 10000:
            continue
        if not (row['price'] > row['ema_8'] > row['ema_21']):
            continue
        clean_sym = row['clean_symbol']
        if clean_sym in fwd_10d:
            analysis_data.append({
                'ema_sep': row['ema_sep_pct'],
                'demarker': row['demarker'],
                'price_dist': row['price_dist_pct'],
                'volume': row['volume'],
                'market_cap': row['market_cap'],
                'forward_return': fwd_10d[clean_sym],
                'is_winner': fwd_10d[clean_sym] > 0,
            })

    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)

        print(f"\nAll power zone stocks with forward returns: {len(analysis_df)}")
        print(f"Overall win rate: {analysis_df['is_winner'].mean()*100:.1f}%")
        print(f"Overall avg return: {analysis_df['forward_return'].mean():+.2f}%")

        # Correlation with returns
        print(f"\nCorrelation with 10-day forward return:")
        for col in ['ema_sep', 'demarker', 'price_dist', 'volume', 'market_cap']:
            corr = analysis_df[col].corr(analysis_df['forward_return'])
            print(f"  {col:15s}: {corr:+.4f}")

        # Winners vs Losers profile
        winners = analysis_df[analysis_df['is_winner']]
        losers = analysis_df[~analysis_df['is_winner']]
        print(f"\nWinners ({len(winners)}) vs Losers ({len(losers)}) profile:")
        for col in ['ema_sep', 'demarker', 'price_dist']:
            w_mean = winners[col].mean()
            l_mean = losers[col].mean()
            diff = w_mean - l_mean
            print(f"  {col:15s}: Winners={w_mean:>+7.3f}  Losers={l_mean:>+7.3f}  Diff={diff:>+7.3f}")

        # DeMarker bucket analysis
        print(f"\nDeMarker bucket analysis:")
        buckets = [(0, 0.20, '<0.20'), (0.20, 0.30, '0.20-0.30'), (0.30, 0.40, '0.30-0.40'),
                   (0.40, 0.50, '0.40-0.50'), (0.50, 0.70, '0.50-0.70'), (0.70, 1.01, '>0.70')]
        for low, high, label in buckets:
            bucket = analysis_df[(analysis_df['demarker'] >= low) & (analysis_df['demarker'] < high)]
            if len(bucket) > 0:
                wr = bucket['is_winner'].mean() * 100
                avg_r = bucket['forward_return'].mean()
                print(f"  DeMarker {label:>10s}: N={len(bucket):>3}  WinRate={wr:>5.1f}%  AvgReturn={avg_r:>+6.2f}%")

        # EMA separation bucket analysis
        print(f"\nEMA separation bucket analysis:")
        sep_buckets = [(0, 0.5, '<0.5%'), (0.5, 1.0, '0.5-1%'), (1.0, 2.0, '1-2%'),
                       (2.0, 3.0, '2-3%'), (3.0, 5.0, '3-5%'), (5.0, 100, '>5%')]
        for low, high, label in sep_buckets:
            bucket = analysis_df[(analysis_df['ema_sep'] >= low) & (analysis_df['ema_sep'] < high)]
            if len(bucket) > 0:
                wr = bucket['is_winner'].mean() * 100
                avg_r = bucket['forward_return'].mean()
                print(f"  EMA Sep {label:>8s}: N={len(bucket):>3}  WinRate={wr:>5.1f}%  AvgReturn={avg_r:>+6.2f}%")

        # Price distance bucket analysis
        print(f"\nPrice distance from EMA8 bucket analysis:")
        dist_buckets = [(0, 1.0, '0-1%'), (1.0, 2.0, '1-2%'), (2.0, 3.0, '2-3%'),
                        (3.0, 5.0, '3-5%'), (5.0, 100, '>5%')]
        for low, high, label in dist_buckets:
            bucket = analysis_df[(analysis_df['price_dist'] >= low) & (analysis_df['price_dist'] < high)]
            if len(bucket) > 0:
                wr = bucket['is_winner'].mean() * 100
                avg_r = bucket['forward_return'].mean()
                print(f"  Dist {label:>6s}: N={len(bucket):>3}  WinRate={wr:>5.1f}%  AvgReturn={avg_r:>+6.2f}%")

    # ===== COMPARISON: CURRENT vs OPTIMAL =====
    print(f"\n{'='*80}")
    print(f"COMPARISON: CURRENT CONFIG vs OPTIMAL CONFIG")
    print(f"{'='*80}")

    current_params = {
        'weights': (40, 30, 30),
        'min_score': 35,
        'quality_thresholds': (0.30, 0.50),
        'accepted_qualities': {'high', 'medium'},
        'min_volume': 50000,
        'min_market_cap': 500,
        'top_n': 50,
    }

    current_bt = run_backtest(entry_df, fwd_10d, current_params)
    optimal_bt = run_backtest(entry_df, fwd_10d, best_params)

    print(f"\n{'Metric':20s} {'Current':>12} {'Optimal':>12} {'Change':>12}")
    print("-" * 60)
    if current_bt and optimal_bt:
        for metric in ['win_rate', 'avg_return', 'median_return', 'profit_factor', 'sharpe', 'n_with_data']:
            curr = current_bt[metric]
            opt = optimal_bt[metric]
            diff = opt - curr
            print(f"{metric:20s} {curr:>12.2f} {opt:>12.2f} {diff:>+12.2f}")

    # Output recommended changes
    print(f"\n{'='*80}")
    print(f"RECOMMENDED PARAMETER UPDATES")
    print(f"{'='*80}")
    print(f"  Scoring weights: {best['weights']}")
    print(f"  Min score threshold: {best['min_score']}")
    print(f"  DeMarker thresholds: high < {best['quality_th'][0]}, medium < {best['quality_th'][1]}")
    print(f"  Accepted qualities: {best['accepted_q']}")
    print(f"  Top N limit: {best['top_n']}")


if __name__ == '__main__':
    main()
