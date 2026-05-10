# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1100.55
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
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.25% | 1.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.25% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 876.65 | 882.44 | 0.00 | ORB-short ORB[879.00,890.00] vol=1.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 879.10 | 882.26 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 853.75 | 859.72 | 0.00 | ORB-short ORB[856.00,867.55] vol=2.0x ATR=3.59 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 857.34 | 858.43 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 867.40 | 862.23 | 0.00 | ORB-long ORB[856.05,864.50] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 864.26 | 863.46 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 858.25 | 863.41 | 0.00 | ORB-short ORB[859.10,869.90] vol=2.7x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:40:00 | 854.38 | 861.05 | 0.00 | T1 1.5R @ 854.38 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 858.25 | 860.54 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 826.70 | 832.56 | 0.00 | ORB-short ORB[830.95,843.15] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 830.56 | 830.10 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:30:00 | 811.20 | 806.71 | 0.00 | ORB-long ORB[800.00,810.50] vol=1.9x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:55:00 | 816.19 | 808.68 | 0.00 | T1 1.5R @ 816.19 |
| Stop hit — per-position SL triggered | 2026-04-07 14:10:00 | 811.20 | 810.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:55:00 | 887.25 | 881.07 | 0.00 | ORB-long ORB[875.10,884.00] vol=1.9x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:55:00 | 893.32 | 884.89 | 0.00 | T1 1.5R @ 893.32 |
| Target hit | 2026-04-15 12:35:00 | 890.00 | 892.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 984.10 | 990.69 | 0.00 | ORB-short ORB[986.00,999.00] vol=1.9x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 988.94 | 990.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:55:00 | 876.65 | 2026-02-12 11:00:00 | 879.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-13 09:30:00 | 853.75 | 2026-02-13 09:40:00 | 857.34 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-17 09:40:00 | 867.40 | 2026-02-17 10:00:00 | 864.26 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 09:50:00 | 858.25 | 2026-02-18 10:40:00 | 854.38 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-18 09:50:00 | 858.25 | 2026-02-18 11:15:00 | 858.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 826.70 | 2026-02-24 10:20:00 | 830.56 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-07 10:30:00 | 811.20 | 2026-04-07 11:55:00 | 816.19 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-07 10:30:00 | 811.20 | 2026-04-07 14:10:00 | 811.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:55:00 | 887.25 | 2026-04-15 10:55:00 | 893.32 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-15 09:55:00 | 887.25 | 2026-04-15 12:35:00 | 890.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-24 09:35:00 | 984.10 | 2026-04-24 09:40:00 | 988.94 | STOP_HIT | 1.00 | -0.49% |
