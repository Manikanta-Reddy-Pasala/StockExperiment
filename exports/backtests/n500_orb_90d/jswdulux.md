# JSW Dulux Ltd. (JSWDULUX)

## Backtest Summary

- **Window:** 2026-04-15 09:15:00 → 2026-05-08 15:25:00 (1275 bars)
- **Last close:** 2950.00
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 3
- **Avg / median % per leg:** 0.11% / 0.22%
- **Sum % (uncompounded):** 1.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.26% | 0.8% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.26% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.05% | 0.4% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.05% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.11% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-04-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:20:00 | 2951.40 | 2967.67 | 0.00 | ORB-short ORB[2953.60,2985.00] vol=2.0x ATR=10.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:35:00 | 2935.50 | 2956.40 | 0.00 | T1 1.5R @ 2935.50 |
| Target hit | 2026-04-17 13:05:00 | 2945.00 | 2944.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 2893.20 | 2897.65 | 0.00 | ORB-short ORB[2894.00,2923.80] vol=3.7x ATR=8.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:00:00 | 2881.12 | 2895.94 | 0.00 | T1 1.5R @ 2881.12 |
| Target hit | 2026-04-21 14:15:00 | 2885.00 | 2880.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 2927.20 | 2940.98 | 0.00 | ORB-short ORB[2928.70,2965.00] vol=3.3x ATR=6.88 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 2934.08 | 2937.66 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:15:00 | 2993.50 | 2983.47 | 0.00 | ORB-long ORB[2947.80,2976.30] vol=2.6x ATR=8.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:20:00 | 3005.60 | 3008.49 | 0.00 | T1 1.5R @ 3005.60 |
| Target hit | 2026-04-27 10:50:00 | 3012.00 | 3012.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 2997.10 | 2982.84 | 0.00 | ORB-long ORB[2970.30,2992.00] vol=7.0x ATR=7.21 |
| Stop hit — per-position SL triggered | 2026-04-29 14:05:00 | 2989.89 | 2992.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 2925.40 | 2943.61 | 0.00 | ORB-short ORB[2952.70,2976.80] vol=1.9x ATR=5.15 |
| Stop hit — per-position SL triggered | 2026-05-06 11:20:00 | 2930.55 | 2943.11 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 2960.00 | 2980.56 | 0.00 | ORB-short ORB[2972.40,3015.60] vol=1.6x ATR=10.73 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 2970.73 | 2979.57 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 2949.90 | 2957.08 | 0.00 | ORB-short ORB[2953.00,2972.30] vol=4.2x ATR=9.12 |
| Stop hit — per-position SL triggered | 2026-05-08 10:25:00 | 2959.02 | 2954.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-17 10:20:00 | 2951.40 | 2026-04-17 11:35:00 | 2935.50 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-17 10:20:00 | 2951.40 | 2026-04-17 13:05:00 | 2945.00 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2026-04-21 10:00:00 | 2893.20 | 2026-04-21 11:00:00 | 2881.12 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-21 10:00:00 | 2893.20 | 2026-04-21 14:15:00 | 2885.00 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2026-04-23 10:40:00 | 2927.20 | 2026-04-23 11:10:00 | 2934.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 10:15:00 | 2993.50 | 2026-04-27 10:20:00 | 3005.60 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-27 10:15:00 | 2993.50 | 2026-04-27 10:50:00 | 3012.00 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-29 11:00:00 | 2997.10 | 2026-04-29 14:05:00 | 2989.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-06 11:10:00 | 2925.40 | 2026-05-06 11:20:00 | 2930.55 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-05-07 11:00:00 | 2960.00 | 2026-05-07 11:15:00 | 2970.73 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-08 09:40:00 | 2949.90 | 2026-05-08 10:25:00 | 2959.02 | STOP_HIT | 1.00 | -0.31% |
