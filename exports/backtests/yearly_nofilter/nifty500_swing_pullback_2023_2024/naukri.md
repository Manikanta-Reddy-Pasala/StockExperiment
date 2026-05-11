# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 964.65
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.65% / -1.18%
- **Sum % (uncompounded):** -2.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.65% | -2.6% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.65% | -2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.65% | -2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 00:00:00 | 925.10 | 798.15 | 879.91 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=23.20 |
| Stop hit — per-position SL triggered | 2023-07-27 00:00:00 | 914.15 | 810.50 | 908.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 00:00:00 | 977.34 | 817.32 | 917.98 | Stage2 pullback-breakout RSI=68 vol=4.1x ATR=24.44 |
| Stop hit — per-position SL triggered | 2023-08-08 00:00:00 | 940.68 | 820.02 | 924.42 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 1049.23 | 881.87 | 1007.25 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=27.03 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 1008.68 | 895.67 | 1027.71 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 1095.11 | 950.26 | 1044.51 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=33.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 1163.02 | 963.51 | 1098.26 | T1 booked 50% @ 1163.02 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-13 00:00:00 | 925.10 | 2023-07-27 00:00:00 | 914.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2023-08-04 00:00:00 | 977.34 | 2023-08-08 00:00:00 | 940.68 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-01-05 00:00:00 | 1049.23 | 2024-01-18 00:00:00 | 1008.68 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1095.11 | 2024-04-08 00:00:00 | 1163.02 | PARTIAL | 0.50 | 6.20% |
