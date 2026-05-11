# United Spirits Ltd. (UNITDSPR)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (662 bars)
- **Last close:** 1280.80
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
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -1.47% / -2.88%
- **Sum % (uncompounded):** -8.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -1.47% | -8.8% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -1.47% | -8.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 5 | 1 | -1.47% | -8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 05:30:00 | 1034.20 | 874.05 | 997.35 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=21.96 |
| Stop hit — per-position SL triggered | 2023-08-28 05:30:00 | 1001.26 | 879.39 | 1001.40 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-09-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 05:30:00 | 1058.70 | 889.03 | 1013.26 | Stage2 pullback-breakout RSI=68 vol=2.7x ATR=22.39 |
| Stop hit — per-position SL triggered | 2023-09-13 05:30:00 | 1025.11 | 896.79 | 1026.55 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-10-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 05:30:00 | 1050.35 | 917.40 | 1021.08 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=22.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 05:30:00 | 1094.64 | 923.45 | 1038.15 | T1 booked 50% @ 1094.64 |
| Stop hit — per-position SL triggered | 2023-10-19 05:30:00 | 1050.35 | 925.98 | 1040.47 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-12-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 05:30:00 | 1084.80 | 966.11 | 1057.81 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=20.80 |
| Stop hit — per-position SL triggered | 2023-12-20 05:30:00 | 1053.60 | 970.71 | 1065.95 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 05:30:00 | 1200.10 | 1051.10 | 1138.30 | Stage2 pullback-breakout RSI=65 vol=5.5x ATR=30.54 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 1154.29 | 1053.37 | 1143.16 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-05-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 05:30:00 | 1231.70 | 1068.75 | 1175.30 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=31.13 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-22 05:30:00 | 1034.20 | 2023-08-28 05:30:00 | 1001.26 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest1 | 2023-09-06 05:30:00 | 1058.70 | 2023-09-13 05:30:00 | 1025.11 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest1 | 2023-10-11 05:30:00 | 1050.35 | 2023-10-17 05:30:00 | 1094.64 | PARTIAL | 0.50 | 4.22% |
| BUY | retest1 | 2023-10-11 05:30:00 | 1050.35 | 2023-10-19 05:30:00 | 1050.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-14 05:30:00 | 1084.80 | 2023-12-20 05:30:00 | 1053.60 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest1 | 2024-04-10 05:30:00 | 1200.10 | 2024-04-15 05:30:00 | 1154.29 | STOP_HIT | 1.00 | -3.82% |
