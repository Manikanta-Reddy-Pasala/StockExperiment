# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 879.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -1.73% / -3.85%
- **Sum % (uncompounded):** -12.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.73% | -12.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.73% | -12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.73% | -12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 782.90 | 671.63 | 753.37 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=20.39 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 752.31 | 676.62 | 761.08 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 768.25 | 694.14 | 758.45 | Stage2 pullback-breakout RSI=53 vol=2.8x ATR=19.83 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 738.50 | 696.58 | 757.43 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 766.50 | 706.04 | 755.97 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=17.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 801.23 | 711.82 | 767.40 | T1 booked 50% @ 801.23 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 788.75 | 714.92 | 779.92 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-18 00:00:00 | 777.90 | 729.94 | 770.68 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=19.96 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 747.96 | 730.90 | 768.00 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 828.85 | 734.12 | 775.36 | Stage2 pullback-breakout RSI=65 vol=3.5x ATR=24.71 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 791.78 | 738.24 | 792.44 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 792.25 | 747.42 | 757.58 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=22.26 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 765.20 | 751.93 | 779.38 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 782.90 | 2023-09-12 00:00:00 | 752.31 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest1 | 2023-10-17 00:00:00 | 768.25 | 2023-10-23 00:00:00 | 738.50 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest1 | 2023-11-17 00:00:00 | 766.50 | 2023-12-01 00:00:00 | 801.23 | PARTIAL | 0.50 | 4.53% |
| BUY | retest1 | 2023-11-17 00:00:00 | 766.50 | 2023-12-06 00:00:00 | 788.75 | STOP_HIT | 0.50 | 2.90% |
| BUY | retest1 | 2024-01-18 00:00:00 | 777.90 | 2024-01-23 00:00:00 | 747.96 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest1 | 2024-02-02 00:00:00 | 828.85 | 2024-02-09 00:00:00 | 791.78 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2024-04-01 00:00:00 | 792.25 | 2024-04-16 00:00:00 | 765.20 | STOP_HIT | 1.00 | -3.41% |
