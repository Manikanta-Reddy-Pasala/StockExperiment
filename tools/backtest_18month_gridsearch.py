#!/usr/bin/env python3
"""
13-Month Full Grid Search Optimization for 8-21 EMA Strategy

Uses all available production data: Jan 2025 → Jan 2026 entry signals,
with forward returns measured Feb 2025 → Feb 2026. (13 rolling windows)

Tests multiple scoring models, weight combos, thresholds, and selection limits
to find the most robust configuration across bull, bear, and sideways markets.
"""

import sys
import os
import subprocess
from datetime import datetime
from collections import defaultdict

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
    m = 2.0 / (span + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append(prices[i] * m + ema[-1] * (1 - m))
    return ema


def calculate_demarker(highs, lows, period=14):
    if len(highs) < period + 1:
        return 0.5
    demax, demin = [], []
    for i in range(1, len(highs)):
        demax.append(max(highs[i] - highs[i-1], 0))
        demin.append(max(-(lows[i] - lows[i-1]), 0))
    if len(demax) < period:
        return 0.5
    s1, s2 = sum(demax[-period:]), sum(demin[-period:])
    if s1 + s2 == 0:
        return 0.5
    return max(0.0, min(1.0, s1 / (s1 + s2)))


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
        indicators.append({
            'symbol': symbol, 'clean_symbol': clean_symbol(symbol),
            'name': meta['name'], 'sector': meta['sector'],
            'price': price, 'ema_8': e8, 'ema_21': e21, 'demarker': demarker,
            'volume': latest['volume'], 'market_cap': meta['market_cap'],
            'ema_sep_pct': ((e8 - e21) / e21 * 100) if e21 > 0 else 0,
            'price_dist_pct': ((price - e8) / e8 * 100) if e8 > 0 else 0,
            'is_etf': is_etf(symbol, meta['name']),
            'is_power_zone': price > e8 > e21,
        })
    return indicators


def process_forward_returns(rows):
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

        ret = {}
        for hd, label in [(5, '5d'), (10, '10d'), (15, '15d')]:
            idx = min(hd, len(data) - 1)
            ret[label] = ((data[idx]['close'] - entry_price) / entry_price) * 100

        all_highs = [d['high'] for d in data[:min(20, len(data))]]
        all_lows = [d['low'] for d in data[:min(20, len(data))]]
        ret['max_gain'] = ((max(all_highs) - entry_price) / entry_price) * 100
        ret['max_loss'] = ((min(all_lows) - entry_price) / entry_price) * 100
        ret['hit_stop_5pct'] = ret['max_loss'] < -5.0

        forward_returns[clean_sym] = ret
    return forward_returns


# ===== SCORING MODELS =====
# Model A: Current (oversold=best for DeMarker)
def score_model_A(price, ema_8, ema_21, demarker, weights):
    if ema_21 <= 0 or price <= 0 or not (price > ema_8 > ema_21):
        return 0.0
    w_sep, w_dm, w_dist = weights

    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    if demarker < 0.20:    dm_score = w_dm * 1.0
    elif demarker < 0.30:  dm_score = w_dm * 0.83
    elif demarker < 0.40:  dm_score = w_dm * 0.60
    elif demarker < 0.50:  dm_score = w_dm * 0.40
    elif demarker < 0.70:  dm_score = w_dm * 0.17
    else:                  dm_score = 0.0

    price_dist_pct = ((price - ema_8) / ema_8) * 100
    if 0 <= price_dist_pct <= 1.0:    dist_score = w_dist * 1.0
    elif price_dist_pct <= 2.0:        dist_score = w_dist * 0.83
    elif price_dist_pct <= 3.0:        dist_score = w_dist * 0.60
    elif price_dist_pct <= 5.0:        dist_score = w_dist * 0.33
    else:                              dist_score = w_dist * 0.10

    return round(sep_score + dm_score + dist_score, 2)


# Model B: Neutral DeMarker (0.40-0.60 is the sweet spot based on 6mo data)
def score_model_B(price, ema_8, ema_21, demarker, weights):
    if ema_21 <= 0 or price <= 0 or not (price > ema_8 > ema_21):
        return 0.0
    w_sep, w_dm, w_dist = weights

    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    # Neutral/mild momentum is best (not oversold, not overbought)
    if 0.40 <= demarker <= 0.60:   dm_score = w_dm * 1.0   # Sweet spot
    elif 0.30 <= demarker < 0.40:  dm_score = w_dm * 0.75
    elif 0.60 < demarker <= 0.70:  dm_score = w_dm * 0.75
    elif demarker < 0.30:          dm_score = w_dm * 0.50   # Too oversold
    else:                          dm_score = 0.0            # Overbought

    price_dist_pct = ((price - ema_8) / ema_8) * 100
    if 0 <= price_dist_pct <= 1.0:    dist_score = w_dist * 1.0
    elif price_dist_pct <= 2.0:        dist_score = w_dist * 0.83
    elif price_dist_pct <= 3.0:        dist_score = w_dist * 0.60
    elif price_dist_pct <= 5.0:        dist_score = w_dist * 0.33
    else:                              dist_score = w_dist * 0.10

    return round(sep_score + dm_score + dist_score, 2)


# Model C: Flat DeMarker (all qualifying DeMarker values equal, focus on EMA+dist)
def score_model_C(price, ema_8, ema_21, demarker, weights):
    if ema_21 <= 0 or price <= 0 or not (price > ema_8 > ema_21):
        return 0.0
    w_sep, w_dm, w_dist = weights

    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    # Flat: all qualifying DeMarker values get same score
    if demarker < 0.70:
        dm_score = w_dm * 1.0
    else:
        dm_score = 0.0

    price_dist_pct = ((price - ema_8) / ema_8) * 100
    if 0 <= price_dist_pct <= 1.0:    dist_score = w_dist * 1.0
    elif price_dist_pct <= 2.0:        dist_score = w_dist * 0.83
    elif price_dist_pct <= 3.0:        dist_score = w_dist * 0.60
    elif price_dist_pct <= 5.0:        dist_score = w_dist * 0.33
    else:                              dist_score = w_dist * 0.10

    return round(sep_score + dm_score + dist_score, 2)


# Model D: Momentum-favoring (higher DeMarker = stronger trend = better)
def score_model_D(price, ema_8, ema_21, demarker, weights):
    if ema_21 <= 0 or price <= 0 or not (price > ema_8 > ema_21):
        return 0.0
    w_sep, w_dm, w_dist = weights

    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    # Momentum: higher DeMarker = stronger buying pressure
    if 0.55 <= demarker <= 0.70:   dm_score = w_dm * 1.0   # Strong momentum
    elif 0.45 <= demarker < 0.55:  dm_score = w_dm * 0.80
    elif 0.35 <= demarker < 0.45:  dm_score = w_dm * 0.50
    elif demarker < 0.35:          dm_score = w_dm * 0.25   # Weak/oversold
    else:                          dm_score = 0.0            # Overbought

    price_dist_pct = ((price - ema_8) / ema_8) * 100
    if 0 <= price_dist_pct <= 1.0:    dist_score = w_dist * 1.0
    elif price_dist_pct <= 2.0:        dist_score = w_dist * 0.83
    elif price_dist_pct <= 3.0:        dist_score = w_dist * 0.60
    elif price_dist_pct <= 5.0:        dist_score = w_dist * 0.33
    else:                              dist_score = w_dist * 0.10

    return round(sep_score + dm_score + dist_score, 2)


SCORING_MODELS = {
    'A_oversold': score_model_A,
    'B_neutral': score_model_B,
    'C_flat_dm': score_model_C,
    'D_momentum': score_model_D,
}


def run_backtest_month(indicators, forward_returns, hold, score_fn, weights,
                       min_score, quality_th, accepted_q, top_n, min_ema_sep=0.0):
    candidates = []
    for ind in indicators:
        if ind['is_etf'] or ind['price'] < 50 or ind['price'] > 10000:
            continue
        if ind['volume'] < 50000 or ind['market_cap'] < 500:
            continue
        if not ind['is_power_zone']:
            continue
        if ind['ema_sep_pct'] < min_ema_sep:
            continue

        score = score_fn(ind['price'], ind['ema_8'], ind['ema_21'],
                         ind['demarker'], weights)

        # Quality check
        dm = ind['demarker']
        if dm < quality_th[0]:
            quality = 'high'
        elif dm < quality_th[1]:
            quality = 'medium'
        else:
            quality = 'low'

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
        if sym in forward_returns and hold in forward_returns[sym]:
            fwd = forward_returns[sym]
            results.append({
                'symbol': sym, 'sector': c.get('sector', ''),
                'score': c['score'], 'quality': c['quality'],
                'demarker': c['demarker'], 'ema_sep': c['ema_sep_pct'],
                'price_dist': c['price_dist_pct'],
                'forward_return': fwd[hold],
                'max_loss': fwd['max_loss'],
                'hit_stop_5pct': fwd['hit_stop_5pct'],
            })

    if not results:
        return None

    returns = [r['forward_return'] for r in results]
    winners = sum(1 for r in returns if r > 0)
    gp = sum(r for r in returns if r > 0)
    gl = abs(sum(r for r in returns if r <= 0))

    return {
        'n_picks': len(results),
        'win_rate': winners / len(returns) * 100,
        'avg_return': sum(returns) / len(returns),
        'total_return': sum(returns),
        'profit_factor': gp / gl if gl > 0 else 99.0,
        'stop_losses': sum(1 for r in results if r['hit_stop_5pct']),
        'results': results,
    }


def main():
    print("=" * 100)
    print("13-MONTH FULL GRID SEARCH OPTIMIZATION - 8-21 EMA Strategy")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Data: Jan 2025 → Jan 2026 entry signals, Feb 2025 → Feb 2026 forward returns")
    print("=" * 100)

    # 13 rolling windows
    windows = []
    for y in range(2025, 2027):
        for m in range(1, 13):
            if y == 2025 and m < 1:
                continue
            if y == 2026 and m > 1:
                break
            fy, fm = (y, m + 1) if m < 12 else (y + 1, 1)
            windows.append((y, m, fy, fm))

    # Cache all data
    all_indicators = {}
    all_fwd_returns = {}

    for entry_y, entry_m, fwd_y, fwd_m in windows:
        label = f"{entry_y}-{entry_m:02d}"
        print(f"  Fetching {label}...", end=" ", flush=True)
        entry_rows = fetch_month_data(entry_y, entry_m)
        fwd_rows = fetch_month_data(fwd_y, fwd_m)

        if entry_rows and fwd_rows:
            all_indicators[label] = process_indicators(entry_rows)
            all_fwd_returns[label] = process_forward_returns(fwd_rows)
            pz = sum(1 for i in all_indicators[label] if i['is_power_zone'])
            print(f"{len(all_indicators[label])} syms ({pz} PZ), {len(all_fwd_returns[label])} fwd")
        else:
            print("SKIP")

    months = sorted(all_indicators.keys())
    n_months = len(months)
    print(f"\nLoaded {n_months} months: {months}")

    # ===== GRID SEARCH =====
    weight_combos = [
        (50, 25, 25),   # Current
        (40, 30, 30),   # Balanced
        (60, 20, 20),   # Heavy EMA sep
        (30, 40, 30),   # DeMarker focus
        (40, 20, 40),   # Sep + Dist
        (30, 30, 40),   # Distance focus
        (70, 15, 15),   # Very heavy EMA sep
        (20, 50, 30),   # Very heavy DeMarker
    ]
    score_thresholds = [15, 20, 25, 30, 35, 40]
    quality_configs = [
        ((0.30, 0.70), {'high', 'medium'}),   # Current (wide)
        ((0.30, 0.50), {'high', 'medium'}),    # Narrow
        ((0.25, 0.65), {'high', 'medium'}),    # Mid-wide
        ((0.30, 0.70), {'high'}),               # High only, wide
    ]
    top_n_options = [5, 8, 10, 15, 20]
    min_ema_sep_options = [0.0, 0.5, 1.0]
    hold_periods = ['5d', '10d', '15d']
    model_names = list(SCORING_MODELS.keys())

    total = (len(model_names) * len(weight_combos) * len(score_thresholds) *
             len(quality_configs) * len(top_n_options) * len(min_ema_sep_options) *
             len(hold_periods))
    print(f"\nTesting {total} parameter combinations across {n_months} months...")
    print(f"  Models: {model_names}")
    print(f"  Weights: {len(weight_combos)}, Scores: {len(score_thresholds)}, "
          f"Quality: {len(quality_configs)}, TopN: {len(top_n_options)}, "
          f"MinSep: {len(min_ema_sep_options)}, Holds: {len(hold_periods)}")

    results_all = []
    tested = 0

    for model_name in model_names:
        score_fn = SCORING_MODELS[model_name]
        for hold in hold_periods:
            for weights in weight_combos:
                for min_score in score_thresholds:
                    for (quality_th, accepted_q) in quality_configs:
                        for top_n in top_n_options:
                            for min_ema_sep in min_ema_sep_options:
                                tested += 1
                                if tested % 2000 == 0:
                                    print(f"  Progress: {tested}/{total}...", flush=True)

                                total_trades = 0
                                total_wins = 0
                                total_return_sum = 0.0
                                total_gp = 0.0
                                total_gl = 0.0
                                total_sl = 0
                                months_positive = 0
                                months_active = 0

                                for month in months:
                                    bt = run_backtest_month(
                                        all_indicators[month], all_fwd_returns[month],
                                        hold, score_fn, weights, min_score,
                                        quality_th, accepted_q, top_n, min_ema_sep
                                    )
                                    if bt and bt['n_picks'] >= 2:
                                        months_active += 1
                                        total_trades += bt['n_picks']
                                        total_wins += int(bt['win_rate'] * bt['n_picks'] / 100)
                                        total_return_sum += bt['total_return']
                                        total_sl += bt['stop_losses']
                                        for r in bt['results']:
                                            if r['forward_return'] > 0:
                                                total_gp += r['forward_return']
                                            else:
                                                total_gl += abs(r['forward_return'])
                                        if bt['avg_return'] > 0:
                                            months_positive += 1

                                if total_trades < 20 or months_active < 5:
                                    continue

                                agg_wr = total_wins / total_trades * 100
                                agg_avg = total_return_sum / total_trades
                                agg_pf = total_gp / total_gl if total_gl > 0 else 99
                                consistency = months_positive / n_months
                                sl_rate = total_sl / total_trades

                                composite = (
                                    agg_wr * 0.30 +
                                    min(agg_pf, 5) * 8.0 +
                                    agg_avg * 5.0 +
                                    consistency * 25.0 -
                                    sl_rate * 15.0
                                )

                                results_all.append({
                                    'model': model_name,
                                    'hold': hold,
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
                                    'sl_pct': round(sl_rate * 100, 1),
                                    'months_active': months_active,
                                    'months_positive': months_positive,
                                    'consistency': round(consistency * 100, 1),
                                    'composite': round(composite, 2),
                                })

    print(f"\nTested {tested} combinations, {len(results_all)} viable\n")

    if not results_all:
        print("ERROR: No viable results!")
        return

    results_all.sort(key=lambda x: x['composite'], reverse=True)

    # ===== TOP 30 =====
    print(f"{'='*140}")
    print("TOP 30 PARAMETER COMBINATIONS (by composite score)")
    print(f"{'='*140}")
    print(f"{'#':>3} {'Model':>12} {'Hold':>4} {'Weights':>12} {'MinSc':>5} {'QTh':>10} "
          f"{'TopN':>4} {'MinSep':>6} | {'WR%':>5} {'AvgR%':>6} {'PF':>5} {'SL%':>5} "
          f"{'Mo':>2}/{n_months} {'Pos':>3} {'Cons%':>5} {'Score':>7} {'Trades':>6}")
    print("─" * 140)

    for i, r in enumerate(results_all[:30], 1):
        print(f"{i:3d} {r['model']:>12} {r['hold']:>4} {str(r['weights']):>12} "
              f"{r['min_score']:>5} {str(r['quality_th']):>10} "
              f"{r['top_n']:>4} {r['min_ema_sep']:>6.1f} | "
              f"{r['win_rate']:>5.1f} {r['avg_return']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['sl_pct']:>5.1f} {r['months_active']:>2}/{n_months} {r['months_positive']:>3} "
              f"{r['consistency']:>5.1f} {r['composite']:>7.2f} {r['total_trades']:>6}")

    # ===== BEST PER MODEL =====
    print(f"\n{'='*100}")
    print("BEST CONFIGURATION PER SCORING MODEL")
    print(f"{'='*100}")
    for model_name in model_names:
        model_results = [r for r in results_all if r['model'] == model_name]
        if model_results:
            best = model_results[0]
            print(f"\n  {model_name}: Hold={best['hold']} W={best['weights']} MinSc={best['min_score']} "
                  f"QTh={best['quality_th']} TopN={best['top_n']} MinSep={best['min_ema_sep']}")
            print(f"    WR={best['win_rate']:.1f}% AvgR={best['avg_return']:+.2f}% PF={best['profit_factor']:.2f} "
                  f"SL={best['sl_pct']:.1f}% Trades={best['total_trades']} "
                  f"Consistency={best['consistency']:.1f}% Composite={best['composite']:.2f}")

    # ===== BEST PER HOLD PERIOD =====
    print(f"\n{'='*100}")
    print("BEST CONFIGURATION PER HOLD PERIOD")
    print(f"{'='*100}")
    for hold in hold_periods:
        hold_results = [r for r in results_all if r['hold'] == hold]
        if hold_results:
            best = hold_results[0]
            print(f"\n  {hold}: Model={best['model']} W={best['weights']} MinSc={best['min_score']} "
                  f"QTh={best['quality_th']} TopN={best['top_n']} MinSep={best['min_ema_sep']}")
            print(f"    WR={best['win_rate']:.1f}% AvgR={best['avg_return']:+.2f}% PF={best['profit_factor']:.2f} "
                  f"SL={best['sl_pct']:.1f}% Trades={best['total_trades']} "
                  f"Consistency={best['consistency']:.1f}% Composite={best['composite']:.2f}")

    # ===== OVERALL BEST - DETAILED =====
    best = results_all[0]
    print(f"\n{'='*100}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*100}")
    print(f"  Scoring Model:   {best['model']}")
    print(f"  Hold Period:     {best['hold']}")
    print(f"  Weights:         {best['weights']} (EMA sep, DeMarker, Price dist)")
    print(f"  Min score:       {best['min_score']}")
    print(f"  Quality thresh:  high < {best['quality_th'][0]}, medium < {best['quality_th'][1]}")
    print(f"  Accepted:        {best['accepted_q']}")
    print(f"  Top N:           {best['top_n']}")
    print(f"  Min EMA Sep:     {best['min_ema_sep']}%")
    print(f"")
    print(f"  PERFORMANCE ({n_months}-month aggregate):")
    print(f"  Total trades:    {best['total_trades']}")
    print(f"  Win rate:        {best['win_rate']:.1f}%")
    print(f"  Avg return:      {best['avg_return']:+.2f}%")
    print(f"  Profit factor:   {best['profit_factor']:.2f}")
    print(f"  Stop loss %:     {best['sl_pct']:.1f}%")
    print(f"  Months active:   {best['months_active']}/{n_months}")
    print(f"  Months positive: {best['months_positive']}/{n_months}")
    print(f"  Consistency:     {best['consistency']:.1f}%")
    print(f"  Composite:       {best['composite']:.2f}")

    # Monthly breakdown for the best config
    best_fn = SCORING_MODELS[best['model']]
    print(f"\n  Monthly Breakdown:")
    all_trades = []
    for month in months:
        bt = run_backtest_month(
            all_indicators[month], all_fwd_returns[month],
            best['hold'], best_fn, best['weights'], best['min_score'],
            best['quality_th'], set(best['accepted_q']), best['top_n'],
            best['min_ema_sep']
        )
        if bt:
            marker = "+" if bt['avg_return'] > 0 else "-"
            print(f"    {month}: Picks={bt['n_picks']:>3}  WR={bt['win_rate']:>5.1f}%  "
                  f"AvgR={bt['avg_return']:>+7.2f}%  PF={bt['profit_factor']:>5.2f}  "
                  f"SL={bt['stop_losses']}/{bt['n_picks']} {marker}")
            for r in bt['results']:
                all_trades.append({**r, 'entry_month': month})
        else:
            print(f"    {month}: No picks")

    # ===== ALL TRADES ANALYSIS =====
    if all_trades:
        print(f"\n{'='*100}")
        print(f"ALL TRADES ANALYSIS ({best['hold']} hold, {n_months} months)")
        print(f"{'='*100}")

        returns = [t['forward_return'] for t in all_trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        sorted_returns = sorted(returns)

        print(f"  Total: {len(all_trades)}  Winners: {len(winners)}  Losers: {len(losers)}")
        print(f"  Win Rate: {len(winners)/len(returns)*100:.1f}%")
        print(f"  Avg Return: {sum(returns)/len(returns):+.2f}%")
        print(f"  Median Return: {sorted_returns[len(sorted_returns)//2]:+.2f}%")
        if winners:
            print(f"  Avg Winner: {sum(winners)/len(winners):+.2f}%")
        if losers:
            print(f"  Avg Loser: {sum(losers)/len(losers):+.2f}%")
        gp = sum(r for r in returns if r > 0)
        gl = abs(sum(r for r in returns if r <= 0))
        print(f"  Profit Factor: {gp/gl:.2f}" if gl > 0 else "  Profit Factor: INF")
        print(f"  Cumulative: {sum(returns):+.2f}%")
        sl5 = sum(1 for t in all_trades if t['hit_stop_5pct'])
        print(f"  Stop Losses (5%): {sl5}/{len(all_trades)} ({sl5/len(all_trades)*100:.1f}%)")

        # Sector analysis
        sector_data = defaultdict(list)
        for t in all_trades:
            sector_data[t.get('sector', 'Unknown') or 'Unknown'].append(t['forward_return'])
        sector_stats = sorted(
            [{'sector': s, 'n': len(r), 'wr': sum(1 for x in r if x > 0)/len(r)*100,
              'avg': sum(r)/len(r)} for s, r in sector_data.items()],
            key=lambda x: x['avg'], reverse=True
        )
        print(f"\n  Sector Performance:")
        for s in sector_stats[:10]:
            print(f"    {s['sector']:30s} N={s['n']:>3}  WR={s['wr']:>5.1f}%  AvgR={s['avg']:>+6.2f}%")

        # DeMarker bucket analysis
        print(f"\n  DeMarker Bucket Analysis:")
        for lo, hi, label in [(0, 0.30, '<0.30'), (0.30, 0.40, '0.30-0.40'),
                               (0.40, 0.50, '0.40-0.50'), (0.50, 0.60, '0.50-0.60'),
                               (0.60, 0.70, '0.60-0.70'), (0.70, 1.01, '>0.70')]:
            bucket = [t for t in all_trades if lo <= t['demarker'] < hi]
            if bucket:
                br = [t['forward_return'] for t in bucket]
                bwr = sum(1 for r in br if r > 0) / len(br) * 100
                bavg = sum(br) / len(br)
                print(f"    DeMarker {label:>10s}: N={len(bucket):>3}  WR={bwr:>5.1f}%  AvgR={bavg:>+6.2f}%")

        # EMA Sep bucket
        print(f"\n  EMA Separation Bucket Analysis:")
        for lo, hi, label in [(0, 0.5, '<0.5%'), (0.5, 1, '0.5-1%'), (1, 2, '1-2%'),
                               (2, 3, '2-3%'), (3, 5, '3-5%'), (5, 100, '>5%')]:
            bucket = [t for t in all_trades if lo <= t['ema_sep'] < hi]
            if bucket:
                br = [t['forward_return'] for t in bucket]
                bwr = sum(1 for r in br if r > 0) / len(br) * 100
                bavg = sum(br) / len(br)
                print(f"    EMA Sep {label:>8s}: N={len(bucket):>3}  WR={bwr:>5.1f}%  AvgR={bavg:>+6.2f}%")

        # Score bucket
        print(f"\n  Score Bucket Analysis:")
        for lo, hi, label in [(0, 30, '<30'), (30, 45, '30-45'), (45, 60, '45-60'),
                               (60, 75, '60-75'), (75, 100, '75+')]:
            bucket = [t for t in all_trades if lo <= t['score'] < hi]
            if bucket:
                br = [t['forward_return'] for t in bucket]
                bwr = sum(1 for r in br if r > 0) / len(br) * 100
                bavg = sum(br) / len(br)
                print(f"    Score {label:>6s}: N={len(bucket):>3}  WR={bwr:>5.1f}%  AvgR={bavg:>+6.2f}%")

    # ===== COMPARE: CURRENT DEPLOYED vs BEST =====
    current_match = None
    for r in results_all:
        if (r['model'] == 'A_oversold' and r['weights'] == (50, 25, 25) and
            r['min_score'] == 25 and r['top_n'] == 10 and
            r['hold'] == '10d' and r['min_ema_sep'] == 0.0 and
            r['quality_th'] == (0.30, 0.70)):
            current_match = r
            break

    if current_match:
        print(f"\n{'='*100}")
        print("CURRENT DEPLOYED vs OVERALL BEST")
        print(f"{'='*100}")
        print(f"{'Metric':25s} {'Current':>12} {'Best':>12} {'Change':>12}")
        print("─" * 65)
        for metric in ['win_rate', 'avg_return', 'profit_factor', 'sl_pct',
                        'consistency', 'total_trades', 'composite']:
            cv = current_match[metric]
            bv = best[metric]
            print(f"{metric:25s} {cv:>12.2f} {bv:>12.2f} {bv-cv:>+12.2f}")
        print(f"\n  Current: Model={current_match['model']} Hold={current_match['hold']} "
              f"W={current_match['weights']} TopN={current_match['top_n']}")
        print(f"  Best:    Model={best['model']} Hold={best['hold']} "
              f"W={best['weights']} TopN={best['top_n']}")
    else:
        print("\n  (Could not find exact current config in grid search results)")

    # Grade
    if best['win_rate'] >= 60 and best['profit_factor'] >= 1.5 and best['consistency'] >= 60:
        grade = "A - STRONG"
    elif best['win_rate'] >= 55 and best['profit_factor'] >= 1.2:
        grade = "B - GOOD"
    elif best['win_rate'] >= 50 and best['profit_factor'] >= 1.0:
        grade = "C - ACCEPTABLE"
    else:
        grade = "D - NEEDS IMPROVEMENT"

    print(f"\n  STRATEGY GRADE: {grade}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
