# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 504.85
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 1 / 2 / 3
- **Avg / median % per leg:** 6.01% / 8.57%
- **Sum % (uncompounded):** 36.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 1 | 2 | 3 | 6.01% | 36.0% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 1 | 2 | 3 | 6.01% | 36.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 1 | 2 | 3 | 6.01% | 36.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 00:00:00 | 1015.00 | 785.00 | 966.09 | Stage2 pullback-breakout RSI=61 vol=4.0x ATR=47.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 00:00:00 | 1109.71 | 802.31 | 1005.00 | T1 booked 50% @ 1109.71 |
| Stop hit — per-position SL triggered | 2024-07-30 00:00:00 | 1024.80 | 812.34 | 1022.06 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 1085.85 | 835.85 | 1024.66 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=51.84 |
| Stop hit — per-position SL triggered | 2024-08-30 00:00:00 | 1114.50 | 858.55 | 1055.81 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 1191.50 | 880.49 | 1080.29 | Stage2 pullback-breakout RSI=67 vol=4.1x ATR=51.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 00:00:00 | 1293.63 | 898.15 | 1146.66 | T1 booked 50% @ 1293.63 |
| Target hit | 2024-10-08 00:00:00 | 1249.30 | 941.51 | 1253.57 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 00:00:00 | 1301.50 | 1032.93 | 1194.01 | Stage2 pullback-breakout RSI=61 vol=4.3x ATR=62.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 00:00:00 | 1427.40 | 1042.72 | 1239.10 | T1 booked 50% @ 1427.40 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-12 00:00:00 | 1015.00 | 2024-07-24 00:00:00 | 1109.71 | PARTIAL | 0.50 | 9.33% |
| BUY | retest1 | 2024-07-12 00:00:00 | 1015.00 | 2024-07-30 00:00:00 | 1024.80 | STOP_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-08-16 00:00:00 | 1085.85 | 2024-08-30 00:00:00 | 1114.50 | STOP_HIT | 1.00 | 2.64% |
| BUY | retest1 | 2024-09-13 00:00:00 | 1191.50 | 2024-09-20 00:00:00 | 1293.63 | PARTIAL | 0.50 | 8.57% |
| BUY | retest1 | 2024-09-13 00:00:00 | 1191.50 | 2024-10-08 00:00:00 | 1249.30 | TARGET_HIT | 0.50 | 4.85% |
| BUY | retest1 | 2024-12-04 00:00:00 | 1301.50 | 2024-12-09 00:00:00 | 1427.40 | PARTIAL | 0.50 | 9.67% |
