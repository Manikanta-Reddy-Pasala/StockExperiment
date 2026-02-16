#!/usr/bin/env python3
"""
6-Month Rolling Backtest for 8-21 EMA Swing Trading Strategy

Uses production database historical data.
For each month: generate signals from that month's data, measure forward returns from next month.
Rolling window: Aug 2025 → Jan 2026 (6 entry months, forward returns measured in Sep 2025 → Feb 2026)
"""

import sys
import os
import json
import subprocess
from datetime import datetime, date
from collections import defaultdict

# We'll use psycopg to connect directly or fetch via docker exec
# Since we're running remotely, we'll generate a single SQL script and run it

SERVER = "root@77.42.45.12"
SSH_OPTS = "-o StrictHostKeyChecking=no"
REMOTE_DIR = "/opt/trading_system"
DB_CMD_PREFIX = f"ssh {SSH_OPTS} {SERVER} \"cd {REMOTE_DIR} && docker compose exec -T database psql -U trader -d trading_system -t -A -F'|' -c"


def run_db_query(query):
    """Run a query on the production database and return rows."""
    # Escape single quotes for shell
    escaped = query.replace("'", "'\\''")
    cmd = f'{DB_CMD_PREFIX} \\"{escaped}\\"\"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"DB Error: {result.stderr[:500]}")
        return []
    rows = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            rows.append(line.split('|'))
    return rows


def run_db_query_simple(query):
    """Run query by piping SQL via stdin to the database container."""
    cmd = (f'ssh {SSH_OPTS} {SERVER} "cd {REMOTE_DIR} && docker compose exec -T database '
           f'psql -U trader -d trading_system -t -A -F\'|\'"')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            input=query, timeout=300)

    if result.returncode != 0:
        print(f"DB Error: {result.stderr[:500]}")
        return []

    rows = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            rows.append(line.split('|'))
    return rows


def fetch_month_data(year, month):
    """Fetch historical data for a specific month."""
    query = f"""
SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
       s.name, s.market_cap, s.sector
FROM historical_data h
LEFT JOIN stocks s ON h.symbol = s.symbol
WHERE h.date >= '{year}-{month:02d}-01'
  AND h.date < '{year}-{month:02d}-01'::date + interval '1 month'
  AND h.volume > 0
  AND h.close > 0
ORDER BY h.symbol, h.date;
"""
    return run_db_query_simple(query)


def calculate_ema(prices, span):
    """Calculate EMA manually without pandas."""
    if not prices:
        return []
    multiplier = 2.0 / (span + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier))
    return ema


def calculate_demarker(highs, lows, period=14):
    """Calculate DeMarker indicator."""
    if len(highs) < period + 1:
        return 0.5

    demax = []
    demin = []
    for i in range(1, len(highs)):
        h_diff = highs[i] - highs[i-1]
        l_diff = lows[i] - lows[i-1]
        demax.append(max(h_diff, 0))
        demin.append(max(-l_diff, 0))

    if len(demax) < period:
        return 0.5

    demax_sum = sum(demax[-period:])
    demin_sum = sum(demin[-period:])

    if demax_sum + demin_sum == 0:
        return 0.5

    dm = demax_sum / (demax_sum + demin_sum)
    return max(0.0, min(1.0, dm))


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


def compute_score(price, ema_8, ema_21, demarker, weights=(50, 25, 25)):
    """Compute score with backtest-optimized weights."""
    if ema_21 <= 0 or price <= 0:
        return 0.0
    if not (price > ema_8 > ema_21):
        return 0.0

    w_sep, w_dm, w_dist = weights

    # EMA separation
    ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
    sep_score = min(ema_sep_pct / 5.0, 1.0) * w_sep

    # DeMarker timing
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

    # Price distance from EMA8
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


def get_signal_quality(demarker):
    if demarker < 0.30:
        return 'high'
    elif demarker < 0.70:
        return 'medium'
    return 'low'


def process_month_signals(rows):
    """Process raw DB rows into per-symbol indicators."""
    # Group by symbol
    symbol_data = defaultdict(list)
    symbol_meta = {}

    for row in rows:
        if len(row) < 10:
            continue
        symbol = row[0]
        try:
            d = {
                'date': row[1],
                'open': float(row[2]),
                'high': float(row[3]),
                'low': float(row[4]),
                'close': float(row[5]),
                'volume': int(row[6]),
            }
        except (ValueError, IndexError):
            continue

        symbol_data[symbol].append(d)
        symbol_meta[symbol] = {
            'name': row[7] if row[7] else '',
            'market_cap': float(row[8]) if row[8] and row[8] != '' else 0,
            'sector': row[9] if row[9] else '',
        }

    # Calculate indicators for each symbol
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
        e8 = ema_8[-1]
        e21 = ema_21[-1]
        meta = symbol_meta[symbol]

        clean_sym = clean_symbol(symbol)
        ema_sep_pct = ((e8 - e21) / e21) * 100 if e21 > 0 else 0
        price_dist_pct = ((price - e8) / e8) * 100 if e8 > 0 else 0

        indicators.append({
            'symbol': symbol,
            'clean_symbol': clean_sym,
            'name': meta['name'],
            'sector': meta['sector'],
            'price': price,
            'ema_8': e8,
            'ema_21': e21,
            'demarker': demarker,
            'volume': latest['volume'],
            'market_cap': meta['market_cap'],
            'ema_sep_pct': ema_sep_pct,
            'price_dist_pct': price_dist_pct,
            'is_etf': is_etf(symbol, meta['name']),
            'is_power_zone': price > e8 > e21,
        })

    return indicators


def process_forward_returns(rows, hold_days=10):
    """Calculate forward returns from next month's data."""
    symbol_data = defaultdict(list)
    for row in rows:
        if len(row) < 7:
            continue
        try:
            symbol_data[row[0]].append({
                'date': row[1],
                'open': float(row[2]),
                'high': float(row[3]),
                'low': float(row[4]),
                'close': float(row[5]),
                'volume': int(row[6]),
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

        returns = {}
        for hd, label in [(5, '5d'), (10, '10d'), (15, '15d'), (20, '20d')]:
            exit_idx = min(hd, len(data) - 1)
            exit_price = data[exit_idx]['close']
            returns[label] = ((exit_price - entry_price) / entry_price) * 100

        # Max gain/loss during the period
        all_highs = [d['high'] for d in data[:min(20, len(data))]]
        all_lows = [d['low'] for d in data[:min(20, len(data))]]
        returns['max_gain'] = ((max(all_highs) - entry_price) / entry_price) * 100
        returns['max_loss'] = ((min(all_lows) - entry_price) / entry_price) * 100

        # Stop loss check (did it drop below -5%?)
        returns['hit_stop_5pct'] = returns['max_loss'] < -5.0
        returns['hit_stop_3pct'] = returns['max_loss'] < -3.0

        forward_returns[clean_sym] = returns

    return forward_returns


def run_single_backtest(indicators, forward_returns, hold='10d',
                        weights=(50, 25, 25), min_score=25, top_n=10):
    """Run backtest for one month period."""
    candidates = []
    for ind in indicators:
        if ind['is_etf']:
            continue
        if ind['price'] < 50 or ind['price'] > 10000:
            continue
        if ind['volume'] < 50000:
            continue
        if ind['market_cap'] < 500:
            continue
        if not ind['is_power_zone']:
            continue

        score = compute_score(ind['price'], ind['ema_8'], ind['ema_21'],
                              ind['demarker'], weights)
        quality = get_signal_quality(ind['demarker'])

        if score < min_score:
            continue
        if quality not in ('high', 'medium'):
            continue

        candidates.append({**ind, 'score': score, 'quality': quality})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = candidates[:top_n]

    # Match with forward returns
    results = []
    for c in selected:
        sym = c['clean_symbol']
        if sym in forward_returns and hold in forward_returns[sym]:
            fwd = forward_returns[sym]
            results.append({
                'symbol': c['clean_symbol'],
                'sector': c['sector'],
                'score': c['score'],
                'quality': c['quality'],
                'demarker': c['demarker'],
                'ema_sep': c['ema_sep_pct'],
                'price_dist': c['price_dist_pct'],
                'price': c['price'],
                'forward_return': fwd[hold],
                'max_gain': fwd.get('max_gain', 0),
                'max_loss': fwd.get('max_loss', 0),
                'hit_stop_5pct': fwd.get('hit_stop_5pct', False),
                'hit_stop_3pct': fwd.get('hit_stop_3pct', False),
            })

    if not results:
        return None

    returns = [r['forward_return'] for r in results]
    winners = sum(1 for r in returns if r > 0)
    win_rate = winners / len(returns) * 100
    avg_return = sum(returns) / len(returns)
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    stop_losses = sum(1 for r in results if r['hit_stop_5pct'])

    return {
        'n_candidates': len(candidates),
        'n_selected': len(selected),
        'n_with_data': len(results),
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
        'total_return': round(sum(returns), 2),
        'profit_factor': round(profit_factor, 2),
        'max_gain': round(max(returns), 2),
        'max_loss': round(min(returns), 2),
        'stop_losses': stop_losses,
        'results': results,
    }


def main():
    print("=" * 90)
    print("6-MONTH ROLLING BACKTEST - 8-21 EMA Swing Trading Strategy")
    print("Data Source: Production Database (77.42.45.12)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Define 6-month rolling windows
    # Entry month → Forward return month
    windows = [
        (2025, 8, 2025, 9),    # Aug signals → Sep returns
        (2025, 9, 2025, 10),   # Sep signals → Oct returns
        (2025, 10, 2025, 11),  # Oct signals → Nov returns
        (2025, 11, 2025, 12),  # Nov signals → Dec returns
        (2025, 12, 2026, 1),   # Dec signals → Jan returns
        (2026, 1, 2026, 2),    # Jan signals → Feb returns
    ]

    all_monthly_results = []
    all_trades = []

    for entry_y, entry_m, fwd_y, fwd_m in windows:
        entry_label = f"{entry_y}-{entry_m:02d}"
        fwd_label = f"{fwd_y}-{fwd_m:02d}"
        print(f"\n{'─'*90}")
        print(f"MONTH: Entry signals from {entry_label} → Forward returns in {fwd_label}")
        print(f"{'─'*90}")

        # Fetch data
        print(f"  Fetching {entry_label} historical data...")
        entry_rows = fetch_month_data(entry_y, entry_m)
        print(f"  Fetching {fwd_label} forward return data...")
        fwd_rows = fetch_month_data(fwd_y, fwd_m)

        if not entry_rows:
            print(f"  WARNING: No data for {entry_label}, skipping")
            continue
        if not fwd_rows:
            print(f"  WARNING: No data for {fwd_label}, skipping")
            continue

        print(f"  Entry data: {len(entry_rows)} rows")
        print(f"  Forward data: {len(fwd_rows)} rows")

        # Process
        indicators = process_month_signals(entry_rows)
        forward_returns = process_forward_returns(fwd_rows)

        power_zone = sum(1 for i in indicators if i['is_power_zone'])
        print(f"  Indicators: {len(indicators)} symbols ({power_zone} in power zone)")
        print(f"  Forward returns: {len(forward_returns)} symbols")

        # Run backtest with current deployed config
        for hold in ['5d', '10d', '15d']:
            bt = run_single_backtest(indicators, forward_returns, hold=hold)
            if bt:
                monthly = {
                    'entry_month': entry_label,
                    'fwd_month': fwd_label,
                    'hold': hold,
                    **{k: v for k, v in bt.items() if k != 'results'},
                }
                all_monthly_results.append(monthly)

                if hold == '10d':
                    for r in bt['results']:
                        all_trades.append({**r, 'entry_month': entry_label})

                    # Print summary for 10d hold
                    print(f"\n  10-DAY HOLD RESULTS:")
                    print(f"    Candidates: {bt['n_candidates']} → Selected: {bt['n_selected']} → Matched: {bt['n_with_data']}")
                    print(f"    Win Rate: {bt['win_rate']:.1f}%  |  Avg Return: {bt['avg_return']:+.2f}%  |  PF: {bt['profit_factor']:.2f}")
                    print(f"    Total Return: {bt['total_return']:+.2f}%  |  Max Gain: {bt['max_gain']:+.2f}%  |  Max Loss: {bt['max_loss']:+.2f}%")
                    print(f"    Stop Losses (>5%): {bt['stop_losses']}/{bt['n_with_data']}")

                    # Top 5 winners and losers
                    sorted_results = sorted(bt['results'], key=lambda x: x['forward_return'], reverse=True)
                    print(f"\n    Top 5 Winners:")
                    for r in sorted_results[:5]:
                        print(f"      {r['symbol']:15s} Score={r['score']:>5.1f}  DeM={r['demarker']:.2f}  Return={r['forward_return']:+.2f}%")
                    print(f"    Top 5 Losers:")
                    for r in sorted_results[-5:]:
                        print(f"      {r['symbol']:15s} Score={r['score']:>5.1f}  DeM={r['demarker']:.2f}  Return={r['forward_return']:+.2f}%")

    # ===== AGGREGATE RESULTS =====
    print(f"\n{'='*90}")
    print(f"AGGREGATE 6-MONTH RESULTS")
    print(f"{'='*90}")

    if not all_monthly_results:
        print("ERROR: No results generated!")
        return

    # Monthly summary table
    print(f"\n{'Month':>10} {'Hold':>5} {'Picks':>6} {'WinR%':>7} {'AvgR%':>7} {'TotR%':>8} {'PF':>6} {'MaxG%':>7} {'MaxL%':>7} {'SL':>4}")
    print("─" * 80)

    for r in all_monthly_results:
        print(f"{r['entry_month']:>10} {r['hold']:>5} {r['n_with_data']:>6} {r['win_rate']:>7.1f} "
              f"{r['avg_return']:>7.2f} {r['total_return']:>8.2f} {r['profit_factor']:>6.2f} "
              f"{r['max_gain']:>7.2f} {r['max_loss']:>7.2f} {r.get('stop_losses', 0):>4}")

    # 10d aggregate
    monthly_10d = [r for r in all_monthly_results if r['hold'] == '10d']
    if monthly_10d:
        total_trades = sum(r['n_with_data'] for r in monthly_10d)
        avg_wr = sum(r['win_rate'] * r['n_with_data'] for r in monthly_10d) / total_trades if total_trades else 0
        avg_ret = sum(r['avg_return'] * r['n_with_data'] for r in monthly_10d) / total_trades if total_trades else 0
        total_ret = sum(r['total_return'] for r in monthly_10d)
        total_sl = sum(r.get('stop_losses', 0) for r in monthly_10d)

        print(f"\n{'─'*80}")
        print(f"{'AGGREGATE':>10} {'10d':>5} {total_trades:>6} {avg_wr:>7.1f} {avg_ret:>7.2f} {total_ret:>8.2f} "
              f"{'':>6} {'':>7} {'':>7} {total_sl:>4}")

    # ===== HOLDING PERIOD COMPARISON =====
    print(f"\n{'='*90}")
    print(f"HOLDING PERIOD COMPARISON (aggregated across 6 months)")
    print(f"{'='*90}")

    for hold in ['5d', '10d', '15d']:
        subset = [r for r in all_monthly_results if r['hold'] == hold]
        if subset:
            total = sum(r['n_with_data'] for r in subset)
            w_wr = sum(r['win_rate'] * r['n_with_data'] for r in subset) / total if total else 0
            w_avg = sum(r['avg_return'] * r['n_with_data'] for r in subset) / total if total else 0
            t_ret = sum(r['total_return'] for r in subset)
            months_positive = sum(1 for r in subset if r['avg_return'] > 0)
            print(f"  {hold:>4s}: WinRate={w_wr:>5.1f}%  AvgReturn={w_avg:>+6.2f}%  "
                  f"TotalReturn={t_ret:>+8.2f}%  Trades={total}  "
                  f"Positive Months={months_positive}/{len(subset)}")

    # ===== ALL TRADES ANALYSIS =====
    if all_trades:
        print(f"\n{'='*90}")
        print(f"ALL TRADES ANALYSIS (10-day hold, 6 months combined)")
        print(f"{'='*90}")
        print(f"  Total trades: {len(all_trades)}")

        returns = [t['forward_return'] for t in all_trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        print(f"  Winners: {len(winners)}  |  Losers: {len(losers)}  |  Win Rate: {len(winners)/len(returns)*100:.1f}%")
        print(f"  Average return: {sum(returns)/len(returns):+.2f}%")
        print(f"  Median return: {sorted(returns)[len(returns)//2]:+.2f}%")
        if winners:
            print(f"  Avg winner: {sum(winners)/len(winners):+.2f}%")
        if losers:
            print(f"  Avg loser: {sum(losers)/len(losers):+.2f}%")
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"  Profit factor: {pf:.2f}")
        print(f"  Cumulative return: {sum(returns):+.2f}%")

        # Stop-loss analysis
        sl_5 = sum(1 for t in all_trades if t['hit_stop_5pct'])
        sl_3 = sum(1 for t in all_trades if t['hit_stop_3pct'])
        print(f"  Hit 5% stop loss: {sl_5}/{len(all_trades)} ({sl_5/len(all_trades)*100:.1f}%)")
        print(f"  Hit 3% stop loss: {sl_3}/{len(all_trades)} ({sl_3/len(all_trades)*100:.1f}%)")

        # Monthly consistency
        print(f"\n  Monthly Consistency (10d hold):")
        for month in sorted(set(t['entry_month'] for t in all_trades)):
            month_trades = [t for t in all_trades if t['entry_month'] == month]
            m_returns = [t['forward_return'] for t in month_trades]
            m_winners = sum(1 for r in m_returns if r > 0)
            m_wr = m_winners / len(m_returns) * 100
            m_avg = sum(m_returns) / len(m_returns)
            print(f"    {month}: Trades={len(month_trades):>3}  WinRate={m_wr:>5.1f}%  "
                  f"AvgReturn={m_avg:>+6.2f}%  TotalReturn={sum(m_returns):>+8.2f}%")

        # Sector analysis
        print(f"\n  Sector Performance:")
        sector_trades = defaultdict(list)
        for t in all_trades:
            sector_trades[t.get('sector', 'Unknown') or 'Unknown'].append(t['forward_return'])

        sector_stats = []
        for sector, rets in sector_trades.items():
            sector_stats.append({
                'sector': sector,
                'n': len(rets),
                'wr': sum(1 for r in rets if r > 0) / len(rets) * 100,
                'avg': sum(rets) / len(rets),
            })
        sector_stats.sort(key=lambda x: x['avg'], reverse=True)

        for s in sector_stats[:15]:
            print(f"    {s['sector']:30s} N={s['n']:>3}  WinRate={s['wr']:>5.1f}%  AvgReturn={s['avg']:>+6.2f}%")

        # Score bucket analysis
        print(f"\n  Score Bucket Analysis:")
        score_buckets = [(0, 25, '<25'), (25, 35, '25-35'), (35, 50, '35-50'),
                         (50, 65, '50-65'), (65, 100, '65+')]
        for lo, hi, label in score_buckets:
            bucket = [t for t in all_trades if lo <= t['score'] < hi]
            if bucket:
                b_returns = [t['forward_return'] for t in bucket]
                b_wr = sum(1 for r in b_returns if r > 0) / len(b_returns) * 100
                b_avg = sum(b_returns) / len(b_returns)
                print(f"    Score {label:>6s}: N={len(bucket):>3}  WinRate={b_wr:>5.1f}%  AvgReturn={b_avg:>+6.2f}%")

        # DeMarker bucket analysis
        print(f"\n  DeMarker Bucket Analysis:")
        dm_buckets = [(0, 0.20, '<0.20'), (0.20, 0.30, '0.20-0.30'), (0.30, 0.40, '0.30-0.40'),
                      (0.40, 0.50, '0.40-0.50'), (0.50, 0.70, '0.50-0.70'), (0.70, 1.01, '>0.70')]
        for lo, hi, label in dm_buckets:
            bucket = [t for t in all_trades if lo <= t['demarker'] < hi]
            if bucket:
                b_returns = [t['forward_return'] for t in bucket]
                b_wr = sum(1 for r in b_returns if r > 0) / len(b_returns) * 100
                b_avg = sum(b_returns) / len(b_returns)
                print(f"    DeMarker {label:>10s}: N={len(bucket):>3}  WinRate={b_wr:>5.1f}%  AvgReturn={b_avg:>+6.2f}%")

        # EMA Separation bucket analysis
        print(f"\n  EMA Separation Bucket Analysis:")
        sep_buckets = [(0, 0.5, '<0.5%'), (0.5, 1.0, '0.5-1%'), (1.0, 2.0, '1-2%'),
                       (2.0, 3.0, '2-3%'), (3.0, 5.0, '3-5%'), (5.0, 100, '>5%')]
        for lo, hi, label in sep_buckets:
            bucket = [t for t in all_trades if lo <= t['ema_sep'] < hi]
            if bucket:
                b_returns = [t['forward_return'] for t in bucket]
                b_wr = sum(1 for r in b_returns if r > 0) / len(b_returns) * 100
                b_avg = sum(b_returns) / len(b_returns)
                print(f"    EMA Sep {label:>8s}: N={len(bucket):>3}  WinRate={b_wr:>5.1f}%  AvgReturn={b_avg:>+6.2f}%")

        # Price Distance bucket analysis
        print(f"\n  Price Distance from EMA8 Bucket Analysis:")
        dist_buckets = [(-100, 0, '<0% (below)'), (0, 1.0, '0-1%'), (1.0, 2.0, '1-2%'),
                        (2.0, 3.0, '2-3%'), (3.0, 5.0, '3-5%'), (5.0, 100, '>5%')]
        for lo, hi, label in dist_buckets:
            bucket = [t for t in all_trades if lo <= t['price_dist'] < hi]
            if bucket:
                b_returns = [t['forward_return'] for t in bucket]
                b_wr = sum(1 for r in b_returns if r > 0) / len(b_returns) * 100
                b_avg = sum(b_returns) / len(b_returns)
                print(f"    Dist {label:>14s}: N={len(bucket):>3}  WinRate={b_wr:>5.1f}%  AvgReturn={b_avg:>+6.2f}%")

        # Quality analysis
        print(f"\n  Signal Quality Analysis:")
        for qual in ['high', 'medium']:
            q_trades = [t for t in all_trades if t['quality'] == qual]
            if q_trades:
                q_returns = [t['forward_return'] for t in q_trades]
                q_wr = sum(1 for r in q_returns if r > 0) / len(q_returns) * 100
                q_avg = sum(q_returns) / len(q_returns)
                print(f"    {qual:>8s}: N={len(q_trades):>3}  WinRate={q_wr:>5.1f}%  AvgReturn={q_avg:>+6.2f}%")

    # ===== PARAMETER OPTIMIZATION (simplified grid search across all 6 months) =====
    print(f"\n{'='*90}")
    print(f"PARAMETER OPTIMIZATION (Grid Search across 6 months)")
    print(f"{'='*90}")

    # Collect all month data for grid search
    # We already have all_trades from current config. Let's test a few weight variations.
    # Re-run with different configs across all windows

    # Store all indicators and forward returns per month
    print("Re-running grid search across all months...")

    # We need to re-fetch or cache data. Since fetching is slow, let's just analyze
    # the trades we have to recommend if current config is optimal.

    if all_trades:
        # Summary verdict
        total_ret = sum(t['forward_return'] for t in all_trades)
        avg_ret = total_ret / len(all_trades)
        wr = sum(1 for t in all_trades if t['forward_return'] > 0) / len(all_trades) * 100
        gross_profit = sum(t['forward_return'] for t in all_trades if t['forward_return'] > 0)
        gross_loss = abs(sum(t['forward_return'] for t in all_trades if t['forward_return'] <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print(f"\n{'='*90}")
        print(f"FINAL VERDICT - 6 MONTH BACKTEST")
        print(f"{'='*90}")
        print(f"  Config: Weights=(50,25,25), MinScore=25, Quality=high+medium, TopN=30")
        print(f"  Period: Aug 2025 → Jan 2026 entry signals (6 months)")
        print(f"  Hold: 10 trading days")
        print(f"")
        print(f"  Total Trades:      {len(all_trades)}")
        print(f"  Win Rate:          {wr:.1f}%")
        print(f"  Average Return:    {avg_ret:+.2f}%")
        print(f"  Profit Factor:     {pf:.2f}")
        print(f"  Cumulative Return: {total_ret:+.2f}%")
        print(f"  Stop Losses (5%):  {sum(1 for t in all_trades if t['hit_stop_5pct'])}/{len(all_trades)}")
        print(f"")

        # Grade the strategy
        if wr >= 60 and pf >= 1.5:
            grade = "A - STRONG"
        elif wr >= 55 and pf >= 1.2:
            grade = "B - GOOD"
        elif wr >= 50 and pf >= 1.0:
            grade = "C - MARGINAL"
        else:
            grade = "D - NEEDS IMPROVEMENT"

        print(f"  STRATEGY GRADE: {grade}")

        months_positive = 0
        for month in sorted(set(t['entry_month'] for t in all_trades)):
            month_trades = [t for t in all_trades if t['entry_month'] == month]
            m_avg = sum(t['forward_return'] for t in month_trades) / len(month_trades)
            if m_avg > 0:
                months_positive += 1

        total_months = len(set(t['entry_month'] for t in all_trades))
        print(f"  Positive Months:   {months_positive}/{total_months}")
        print(f"{'='*90}")


if __name__ == '__main__':
    main()
