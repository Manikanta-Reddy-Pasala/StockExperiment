# NATCO Pharma Ltd. (NATCOPHARM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1174.90
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 2
- **Avg / median % per leg:** 0.02% / -0.31%
- **Sum % (uncompounded):** 0.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.02% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 858.80 | 853.22 | 0.00 | ORB-long ORB[848.65,857.25] vol=2.5x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-02-11 10:55:00 | 856.13 | 853.58 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:55:00 | 857.10 | 850.18 | 0.00 | ORB-long ORB[842.25,853.50] vol=3.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-02-12 10:10:00 | 853.81 | 851.88 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 957.70 | 951.43 | 0.00 | ORB-long ORB[945.00,955.00] vol=2.3x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-03-18 09:40:00 | 953.86 | 952.29 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 1112.50 | 1106.97 | 0.00 | ORB-long ORB[1096.85,1112.00] vol=1.9x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 1107.75 | 1107.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 1104.40 | 1096.72 | 0.00 | ORB-long ORB[1087.00,1099.95] vol=2.7x ATR=4.62 |
| Stop hit — per-position SL triggered | 2026-04-30 09:35:00 | 1099.78 | 1097.40 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 1107.60 | 1105.27 | 0.00 | ORB-long ORB[1093.10,1104.80] vol=5.6x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:40:00 | 1114.65 | 1110.24 | 0.00 | T1 1.5R @ 1114.65 |
| Target hit | 2026-05-04 11:50:00 | 1120.50 | 1120.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-05-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:25:00 | 1154.00 | 1141.69 | 0.00 | ORB-long ORB[1131.80,1148.00] vol=3.3x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 1162.23 | 1147.80 | 0.00 | T1 1.5R @ 1162.23 |
| Stop hit — per-position SL triggered | 2026-05-06 14:45:00 | 1154.00 | 1155.08 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1173.00 | 1161.74 | 0.00 | ORB-long ORB[1152.00,1169.00] vol=2.5x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 1168.62 | 1162.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:40:00 | 858.80 | 2026-02-11 10:55:00 | 856.13 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-12 09:55:00 | 857.10 | 2026-02-12 10:10:00 | 853.81 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 09:30:00 | 957.70 | 2026-03-18 09:40:00 | 953.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-29 09:40:00 | 1112.50 | 2026-04-29 09:45:00 | 1107.75 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-30 09:30:00 | 1104.40 | 2026-04-30 09:35:00 | 1099.78 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-04 09:30:00 | 1107.60 | 2026-05-04 09:40:00 | 1114.65 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-05-04 09:30:00 | 1107.60 | 2026-05-04 11:50:00 | 1120.50 | TARGET_HIT | 0.50 | 1.16% |
| BUY | retest1 | 2026-05-06 10:25:00 | 1154.00 | 2026-05-06 11:00:00 | 1162.23 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-05-06 10:25:00 | 1154.00 | 2026-05-06 14:45:00 | 1154.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:30:00 | 1173.00 | 2026-05-07 10:35:00 | 1168.62 | STOP_HIT | 1.00 | -0.37% |
