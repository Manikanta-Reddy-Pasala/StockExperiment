# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1051.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 7
- **Avg / median % per leg:** 0.34% / 0.34%
- **Sum % (uncompounded):** 5.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 3 | 4 | 7 | 0.38% | 5.3% |
| BUY @ 2nd Alert (retest1) | 14 | 10 | 71.4% | 3 | 4 | 7 | 0.38% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.25% | -0.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.25% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 10 | 66.7% | 3 | 5 | 7 | 0.34% | 5.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 993.00 | 985.93 | 0.00 | ORB-long ORB[980.30,990.25] vol=1.8x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:40:00 | 996.40 | 988.27 | 0.00 | T1 1.5R @ 996.40 |
| Stop hit — per-position SL triggered | 2026-02-12 11:50:00 | 993.00 | 988.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:55:00 | 1000.00 | 993.67 | 0.00 | ORB-long ORB[985.85,993.10] vol=1.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:15:00 | 1002.82 | 994.78 | 0.00 | T1 1.5R @ 1002.82 |
| Stop hit — per-position SL triggered | 2026-02-16 12:10:00 | 1000.00 | 997.10 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 1031.15 | 1025.07 | 0.00 | ORB-long ORB[1019.00,1025.90] vol=1.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 1035.07 | 1029.34 | 0.00 | T1 1.5R @ 1035.07 |
| Stop hit — per-position SL triggered | 2026-02-19 10:35:00 | 1031.15 | 1031.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 1021.35 | 1019.84 | 0.00 | ORB-long ORB[1009.20,1018.75] vol=2.2x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:45:00 | 1024.84 | 1020.05 | 0.00 | T1 1.5R @ 1024.84 |
| Target hit | 2026-02-20 15:20:00 | 1029.15 | 1026.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 963.30 | 959.93 | 0.00 | ORB-long ORB[949.75,962.90] vol=2.3x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:10:00 | 967.59 | 960.90 | 0.00 | T1 1.5R @ 967.59 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 963.30 | 961.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1017.30 | 1010.19 | 0.00 | ORB-long ORB[997.65,1008.15] vol=4.3x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:40:00 | 1022.30 | 1015.22 | 0.00 | T1 1.5R @ 1022.30 |
| Target hit | 2026-04-21 11:15:00 | 1029.45 | 1031.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:10:00 | 1049.35 | 1044.13 | 0.00 | ORB-long ORB[1035.30,1046.80] vol=3.9x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:20:00 | 1054.68 | 1048.02 | 0.00 | T1 1.5R @ 1054.68 |
| Target hit | 2026-04-23 15:20:00 | 1055.05 | 1050.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1005.60 | 1017.08 | 0.00 | ORB-short ORB[1013.20,1024.40] vol=4.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-05-06 11:30:00 | 1008.10 | 1016.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 10:55:00 | 993.00 | 2026-02-12 11:40:00 | 996.40 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-12 10:55:00 | 993.00 | 2026-02-12 11:50:00 | 993.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:55:00 | 1000.00 | 2026-02-16 11:15:00 | 1002.82 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-16 10:55:00 | 1000.00 | 2026-02-16 12:10:00 | 1000.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:35:00 | 1031.15 | 2026-02-19 09:50:00 | 1035.07 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-19 09:35:00 | 1031.15 | 2026-02-19 10:35:00 | 1031.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:40:00 | 1021.35 | 2026-02-20 10:45:00 | 1024.84 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-20 10:40:00 | 1021.35 | 2026-02-20 15:20:00 | 1029.15 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2026-03-05 10:40:00 | 963.30 | 2026-03-05 11:10:00 | 967.59 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-05 10:40:00 | 963.30 | 2026-03-05 11:45:00 | 963.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1017.30 | 2026-04-21 09:40:00 | 1022.30 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1017.30 | 2026-04-21 11:15:00 | 1029.45 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2026-04-23 10:10:00 | 1049.35 | 2026-04-23 12:20:00 | 1054.68 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-23 10:10:00 | 1049.35 | 2026-04-23 15:20:00 | 1055.05 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-06 11:15:00 | 1005.60 | 2026-05-06 11:30:00 | 1008.10 | STOP_HIT | 1.00 | -0.25% |
