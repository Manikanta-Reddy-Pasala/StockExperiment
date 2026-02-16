#!/usr/bin/env python3
"""
6-Month Grid Search Optimization for 8-21 EMA Strategy

Caches all monthly data, then tests parameter combinations across all 6 months.
Finds optimal config that works consistently across different market conditions.
"""

import sys
import os
import subprocess
from datetime import datetime
from collections import defaultdict
from itertools import product

SERVER = "root@77.42.45.12"
SSH_OPTS = "-o StrictHostKeyChecking=no"
REMOTE_DIR = "/opt/trading_system"

ETF_EXCLUSION_SYMBOLS = {
    'LIQUIDBEES', 'NIFTYBEES', 'BANKBEES', 'MON100', 'GOLDBEES',
    'SILVERBEES', 'JUNIORBEES', 'ITBEES', 'PSUBNKBEES', 'SETFNIF50',
    'SETFNIFBK', 'NETFIT', 'HABOREES', 'HNGSNGBEES', 'SHARIABEES',
    'DIVOPPBEES', 'INFRABEES', 'CPSEETF', 'CONSUMBEES', 'HEALTHY',
    'MOM100', 'MOM50', 'PHARMABEES', 'AUTOBEES', 'COMMOETF',
}


def run_db_query(query):
    cmd = (f'ssh {SSH_OPTS} {SERVER} "cd {REMOTE_DIR} && docker compose exec -T database '
           f'psql -U trader -d trading_system -t -A -F\'|\'"')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            input=query, timeout=300)
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            rows.append(line.split('|'))
    return rows


def clean_symbol(raw):
    return raw.upper().replace('NSE:', '').replace('BSE:', '').replace('-EQ', '').strip()


def is_etf(symbol, name):
    clean = clean_symbol(symbol)
    n = (name or '').upper()
    return 'ETF' in n or 'ETF' in clean or 'BEES' in clean or clean in ETF_EXCLUSION_SYMBOLS


def calculate_ema(prices, span):
    if not prices:
        return []
    multiplier = 2.0 / (span + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier))
    return ema


def calculate_demarker(highs, lows, period=14):
    if len(highs) < period + 1:
        return 0.5
    demax = []
    demin = []
    for i in range(1, len(highs)):
        demax.append(max(highs[i] - highs[i-1], 0))
        demin.append(max(-(lows[i] - lows[i-1]), 0))
    if len(demax) < period:
        return 0.5
    demax_sum = sum(demax[-period:])
    demin_sum = sum(demin[-period:])
    if demax_sum + demin_sum == 0:
        return 0.5
    return max(0.0, min(1.0, demax_sum / (demax_sum + demin_sum)))


def fetch_month_data(year, month):
    query = f"""
SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
       s.name, s.market_cap, s.sector
FROM historical_data h
LEFT JOIN stocks s ON h.symbol = s.symbol
WHERE h.date >= '{year}-{month:02d}-01'
  AND h.date < '{year}-{month:02d}-01'::date + interval '1 month'
  AND h.volume > 0 AND h.close > 0
ORDER BY h.symbol, h.date;
"""
    return run_db_query(query)


def process_indicators(rows):
    symbol_data = defaultdict(list)
    symbol_meta = {}
    for row in rows:
        if len(row) < 10:
            continue
        symbol = row[0]
        try:
            symbol_data[symbol].append({
                'date': row[1], 'open': float(row[2]), 'high': float(row[3]),
                'low': float(row[4]), 'close': float(row[5]), 'volume': int(row[6]),
            })
        except (ValueError, IndexError):
            continue
        symbol_meta[symbol] = {
            'name': row[7] or '', 'market_cap': float(row[8]) if row[8] else 0,
            'sector': row[9] or '',
        }

    indicators = []
    for symbol, data in symbol_data.items():
        if len(data) < 14:
            continue
        data.sort(key=lambda x: x['date'])
        closes = [d['close'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        ema_8 = calculate_ema(closes, 8)
        ema_21 = calculate_ema(closes, 21)
        demarker = calculate_demarker(highs, lows)
        latest = data[-1]
        price = latest['close']
        e8, e21 = ema_8[-1], ema_21[-1]
        meta = symbol_meta[symbol]
        clean_sym = clean_symbol(symbol)
        indicators.append({
            'symbol': symbol, 'clean_symbol': clean_sym,
            'name': meta['name'], 'sector': meta['sector'],
            'price': price, 'ema_8': e8, 'ema_21': e21, 'demarker': demarker,
            'volume': latest['volume'], 'market_cap': meta['market_cap'],
            'ema_sep_pct': ((e8 - e21) / e21 * 100) if e21 > 0 else 0,
            'price_dist_pct': ((price - e8) / e8 * 100) if e8 > 0 else 0,
            'is_etf': is_etf(symbol, meta['name']),
            'is_power_zone': price > e8 > e21,
        })
    return indicators


def process_forward_returns(rows, hold_days=10):
    symbol_data = defaultdict(list)
    for row in rows:
        if len(row) < 7:
            continue
        try:
            symbol_data[row[0]].append({
                'date': row[1], 'open': float(row[2]), 'high': float(row[3]),
                'low': float(row[4]), 'close': float(row[5]), 'volume': int(row[6]),
            })
        except (ValueError, IndexError):
            continue

    forward_returns = {}
    for symbol, data in symbol_data.items():
        if len(data) < 5:
            continue
        data.sort(key=lambda x: x['date'])
        entry_price = data[0]['close']
        clean_sym = clean_symbol(symbol)

        exit_idx = min(hold_days, len(data) - 1)
        fwd_ret = ((data[exit_idx]['close'] - entry_price) / entry_price) * 100

        all_highs = [d['high'] for d in data[:min(20, len(data))]]
        all_lows = [d['low'] for d in data[:min(20, len(data))]]
        max_gain = ((max(all_highs) - entry_price) / entry_price) * 100
        max_loss = ((min(all_lows) - entry_price) / entry_price) * 100

        forward_returns[clean_sym] = {
            'return': fwd_ret, 'max_gain': max_gain, 'max_loss': max_loss,
            'hit_stop_5pct': max_loss < -5.0,
        }
    return forward_returns


def compute_score(price, ema_8, ema_21, demarker, weights, ema_sep_cap=5.0):
    if ema_21 <= 0 or price <= 0 or not (price > ema_8 > ema_21):
        return 0.0
    w_sep, w_dm, w_dist = weights

    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / ema_sep_cap, 1.0) * w_sep

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


def get_signal_quality(demarker, thresholds):
    if demarker < thresholds[0]:
        return 'high'
    elif demarker < thresholds[1]:
        return 'medium'
    return 'low'


def run_backtest_month(indicators, forward_returns, weights, min_score, quality_th,
                       accepted_q, top_n, min_ema_sep=0.0, min_vol=50000, min_mcap=500):
    candidates = []
    for ind in indicators:
        if ind['is_etf'] or ind['price'] < 50 or ind['price'] > 10000:
            continue
        if ind['volume'] < min_vol or ind['market_cap'] < min_mcap:
            continue
        if not ind['is_power_zone']:
            continue
        if ind['ema_sep_pct'] < min_ema_sep:
            continue

        score = compute_score(ind['price'], ind['ema_8'], ind['ema_21'],
                              ind['demarker'], weights)
        quality = get_signal_quality(ind['demarker'], quality_th)

        if score < min_score or quality not in accepted_q:
            continue

        candidates.append({**ind, 'score': score, 'quality': quality})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = candidates[:top_n]

    results = []
    for c in selected:
        sym = c['clean_symbol']
        if sym in forward_returns:
            fwd = forward_returns[sym]
            results.append({
                'symbol': sym, 'score': c['score'], 'quality': c['quality'],
                'demarker': c['demarker'], 'ema_sep': c['ema_sep_pct'],
                'price_dist': c['price_dist_pct'],
                'forward_return': fwd['return'],
                'max_loss': fwd['max_loss'],
                'hit_stop_5pct': fwd['hit_stop_5pct'],
            })

    if not results:
        return None

    returns = [r['forward_return'] for r in results]
    winners = sum(1 for r in returns if r > 0)
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r <= 0))

    return {
        'n_picks': len(results),
        'win_rate': winners / len(returns) * 100,
        'avg_return': sum(returns) / len(returns),
        'total_return': sum(returns),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.0,
        'stop_losses': sum(1 for r in results if r['hit_stop_5pct']),
        'results': results,
    }


def main():
    print("=" * 90)
    print("6-MONTH GRID SEARCH OPTIMIZATION")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Define windows
    windows = [
        (2025, 8, 2025, 9),
        (2025, 9, 2025, 10),
        (2025, 10, 2025, 11),
        (2025, 11, 2025, 12),
        (2025, 12, 2026, 1),
        (2026, 1, 2026, 2),
    ]

    # Cache all data
    all_indicators = {}
    all_fwd_returns = {}

    for entry_y, entry_m, fwd_y, fwd_m in windows:
        label = f"{entry_y}-{entry_m:02d}"
        print(f"  Fetching {label} data...", end=" ", flush=True)
        entry_rows = fetch_month_data(entry_y, entry_m)
        fwd_rows = fetch_month_data(fwd_y, fwd_m)

        if entry_rows and fwd_rows:
            all_indicators[label] = process_indicators(entry_rows)
            all_fwd_returns[label] = process_forward_returns(fwd_rows)
            pz = sum(1 for i in all_indicators[label] if i['is_power_zone'])
            print(f"{len(all_indicators[label])} syms ({pz} PZ), {len(all_fwd_returns[label])} fwd")
        else:
            print("SKIP - no data")

    months = sorted(all_indicators.keys())
    print(f"\nLoaded {len(months)} months: {months}")

    # ===== GRID SEARCH =====
    weight_combos = [
        (50, 25, 25),   # Current
        (40, 30, 30),   # Balanced
        (30, 40, 30),   # DeMarker focus
        (30, 30, 40),   # Distance focus
        (60, 20, 20),   # Heavy EMA sep
        (20, 40, 40),   # DM + Dist focus
        (40, 20, 40),   # Sep + Dist
        (35, 35, 30),   # Near-equal
    ]
    score_thresholds = [15, 20, 25, 30, 35, 40, 45]
    quality_configs = [
        ((0.30, 0.50), {'high', 'medium'}),
        ((0.25, 0.45), {'high', 'medium'}),
        ((0.30, 0.70), {'high', 'medium'}),
        ((0.30, 0.50), {'high'}),
        ((0.20, 0.40), {'high', 'medium'}),
    ]
    top_n_options = [10, 15, 20, 30]
    min_ema_sep_options = [0.0, 0.5, 1.0]

    total = (len(weight_combos) * len(score_thresholds) * len(quality_configs) *
             len(top_n_options) * len(min_ema_sep_options))
    print(f"\nTesting {total} parameter combinations across {len(months)} months...")

    results_all = []
    tested = 0

    for weights in weight_combos:
        for min_score in score_thresholds:
            for (quality_th, accepted_q) in quality_configs:
                for top_n in top_n_options:
                    for min_ema_sep in min_ema_sep_options:
                        tested += 1
                        if tested % 500 == 0:
                            print(f"  Progress: {tested}/{total}...", flush=True)

                        month_results = []
                        total_trades = 0
                        total_wins = 0
                        total_return_sum = 0.0
                        total_gross_profit = 0.0
                        total_gross_loss = 0.0
                        total_sl = 0
                        months_positive = 0

                        for month in months:
                            bt = run_backtest_month(
                                all_indicators[month], all_fwd_returns[month],
                                weights, min_score, quality_th, accepted_q,
                                top_n, min_ema_sep
                            )
                            if bt and bt['n_picks'] >= 2:
                                month_results.append(bt)
                                total_trades += bt['n_picks']
                                total_wins += int(bt['win_rate'] * bt['n_picks'] / 100)
                                total_return_sum += bt['total_return']
                                total_sl += bt['stop_losses']
                                for r in bt['results']:
                                    if r['forward_return'] > 0:
                                        total_gross_profit += r['forward_return']
                                    else:
                                        total_gross_loss += abs(r['forward_return'])
                                if bt['avg_return'] > 0:
                                    months_positive += 1

                        if total_trades < 10 or len(month_results) < 3:
                            continue

                        agg_wr = total_wins / total_trades * 100
                        agg_avg = total_return_sum / total_trades
                        agg_pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 99

                        # Composite score: win rate, profit factor, consistency, avg return
                        # Penalize heavy for losing months, reward consistency
                        consistency = months_positive / len(months)
                        composite = (
                            agg_wr * 0.25 +
                            min(agg_pf, 5) * 8.0 +
                            agg_avg * 5.0 +
                            consistency * 20.0 -
                            (total_sl / total_trades) * 15.0  # penalize stop losses
                        )

                        results_all.append({
                            'weights': weights,
                            'min_score': min_score,
                            'quality_th': quality_th,
                            'accepted_q': tuple(sorted(accepted_q)),
                            'top_n': top_n,
                            'min_ema_sep': min_ema_sep,
                            'total_trades': total_trades,
                            'win_rate': round(agg_wr, 1),
                            'avg_return': round(agg_avg, 2),
                            'total_return': round(total_return_sum, 2),
                            'profit_factor': round(agg_pf, 2),
                            'stop_loss_pct': round(total_sl / total_trades * 100, 1),
                            'months_active': len(month_results),
                            'months_positive': months_positive,
                            'consistency': round(consistency * 100, 1),
                            'composite': round(composite, 2),
                        })

    print(f"\nTested {tested} combinations, {len(results_all)} viable")

    if not results_all:
        print("ERROR: No viable results!")
        return

    # Sort by composite score
    results_all.sort(key=lambda x: x['composite'], reverse=True)

    # ===== TOP 30 RESULTS =====
    print(f"\n{'='*120}")
    print("TOP 30 PARAMETER COMBINATIONS (by composite score)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Weights':>12} {'MinSc':>5} {'QTh':>10} {'Qual':>10} {'TopN':>4} {'MinSep':>6} | "
          f"{'WR%':>5} {'AvgR%':>6} {'PF':>5} {'SL%':>5} {'Mos':>3} {'Pos':>3} {'Cons%':>5} {'Score':>7} {'Trades':>6}")
    print("─" * 120)

    for i, r in enumerate(results_all[:30], 1):
        print(f"{i:3d} {str(r['weights']):>12} {r['min_score']:>5} {str(r['quality_th']):>10} "
              f"{str(r['accepted_q']):>10} {r['top_n']:>4} {r['min_ema_sep']:>6.1f} | "
              f"{r['win_rate']:>5.1f} {r['avg_return']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['stop_loss_pct']:>5.1f} {r['months_active']:>3} {r['months_positive']:>3} "
              f"{r['consistency']:>5.1f} {r['composite']:>7.2f} {r['total_trades']:>6}")

    # ===== BEST CONFIG DETAILS =====
    best = results_all[0]
    print(f"\n{'='*90}")
    print("BEST CONFIGURATION")
    print(f"{'='*90}")
    print(f"  Weights:         {best['weights']} (EMA sep, DeMarker, Price dist)")
    print(f"  Min score:       {best['min_score']}")
    print(f"  Quality thresh:  high < {best['quality_th'][0]}, medium < {best['quality_th'][1]}")
    print(f"  Accepted:        {best['accepted_q']}")
    print(f"  Top N:           {best['top_n']}")
    print(f"  Min EMA Sep:     {best['min_ema_sep']}%")
    print(f"")
    print(f"  PERFORMANCE (6-month aggregate):")
    print(f"  Total trades:    {best['total_trades']}")
    print(f"  Win rate:        {best['win_rate']:.1f}%")
    print(f"  Avg return:      {best['avg_return']:+.2f}%")
    print(f"  Profit factor:   {best['profit_factor']:.2f}")
    print(f"  Stop loss %:     {best['stop_loss_pct']:.1f}%")
    print(f"  Months active:   {best['months_active']}/{len(months)}")
    print(f"  Months positive: {best['months_positive']}/{len(months)}")
    print(f"  Consistency:     {best['consistency']:.1f}%")

    # Run the best config month by month for detail
    print(f"\n  Monthly Breakdown:")
    best_weights = best['weights']
    best_quality_th = best['quality_th']
    best_accepted_q = set(best['accepted_q'])

    for month in months:
        bt = run_backtest_month(
            all_indicators[month], all_fwd_returns[month],
            best_weights, best['min_score'], best_quality_th,
            best_accepted_q, best['top_n'], best['min_ema_sep']
        )
        if bt:
            print(f"    {month}: Picks={bt['n_picks']:>3}  WR={bt['win_rate']:>5.1f}%  "
                  f"AvgR={bt['avg_return']:>+6.2f}%  PF={bt['profit_factor']:>5.2f}  "
                  f"SL={bt['stop_losses']}/{bt['n_picks']}")
        else:
            print(f"    {month}: No picks")

    # ===== COMPARE CURRENT vs BEST =====
    # Find current config in results
    current_match = None
    for r in results_all:
        if (r['weights'] == (50, 25, 25) and r['min_score'] == 25 and
                r['min_ema_sep'] == 0.0 and r['top_n'] == 30):
            current_match = r
            break

    if current_match:
        print(f"\n{'='*90}")
        print("CURRENT CONFIG vs BEST CONFIG")
        print(f"{'='*90}")
        print(f"{'Metric':25s} {'Current':>12} {'Best':>12} {'Change':>12}")
        print("─" * 65)
        for metric in ['win_rate', 'avg_return', 'profit_factor', 'stop_loss_pct',
                        'consistency', 'total_trades', 'composite']:
            curr = current_match[metric]
            opt = best[metric]
            diff = opt - curr
            print(f"{metric:25s} {curr:>12.2f} {opt:>12.2f} {diff:>+12.2f}")

    # ===== STABILITY ANALYSIS: Top 10 configs performance per month =====
    print(f"\n{'='*90}")
    print("STABILITY: Top 5 configs across all months")
    print(f"{'='*90}")

    for rank, cfg in enumerate(results_all[:5], 1):
        print(f"\n  Config #{rank}: W={cfg['weights']} MinSc={cfg['min_score']} "
              f"MinSep={cfg['min_ema_sep']} TopN={cfg['top_n']}")
        monthly_wrs = []
        for month in months:
            bt = run_backtest_month(
                all_indicators[month], all_fwd_returns[month],
                cfg['weights'], cfg['min_score'], cfg['quality_th'],
                set(cfg['accepted_q']), cfg['top_n'], cfg['min_ema_sep']
            )
            if bt:
                monthly_wrs.append(bt['win_rate'])
                marker = "+" if bt['avg_return'] > 0 else "-"
                print(f"    {month}: WR={bt['win_rate']:>5.1f}%  AvgR={bt['avg_return']:>+6.2f}% {marker}")
            else:
                print(f"    {month}: No picks")
        if monthly_wrs:
            wr_std = (sum((w - sum(monthly_wrs)/len(monthly_wrs))**2 for w in monthly_wrs) / len(monthly_wrs)) ** 0.5
            print(f"    WR Std Dev: {wr_std:.1f}%  (lower = more consistent)")

    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
