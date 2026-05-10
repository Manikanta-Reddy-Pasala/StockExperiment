# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 206.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 9
- **Avg / median % per leg:** 0.16% / 0.29%
- **Sum % (uncompounded):** 4.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.21% | 2.3% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.21% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.12% | 1.7% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.12% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 13 | 52.0% | 4 | 12 | 9 | 0.16% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 209.99 | 211.24 | 0.00 | ORB-short ORB[212.30,215.44] vol=1.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-02-09 10:55:00 | 211.28 | 211.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 211.03 | 211.85 | 0.00 | ORB-short ORB[211.26,212.70] vol=1.7x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:55:00 | 210.04 | 211.24 | 0.00 | T1 1.5R @ 210.04 |
| Target hit | 2026-02-10 15:20:00 | 209.82 | 210.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 210.00 | 210.05 | 0.00 | ORB-short ORB[210.13,211.74] vol=12.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 209.39 | 210.04 | 0.00 | T1 1.5R @ 209.39 |
| Stop hit — per-position SL triggered | 2026-02-11 13:40:00 | 210.00 | 209.97 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 205.30 | 204.42 | 0.00 | ORB-long ORB[203.00,204.82] vol=2.3x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:40:00 | 206.17 | 204.99 | 0.00 | T1 1.5R @ 206.17 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 205.30 | 205.43 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 201.95 | 202.99 | 0.00 | ORB-short ORB[203.32,205.17] vol=2.5x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-18 12:05:00 | 202.45 | 202.75 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 201.31 | 201.91 | 0.00 | ORB-short ORB[201.70,203.44] vol=3.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-19 11:30:00 | 201.67 | 201.88 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 11:05:00 | 185.39 | 181.52 | 0.00 | ORB-long ORB[178.20,180.25] vol=3.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-03-04 11:25:00 | 184.54 | 182.33 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 178.49 | 179.57 | 0.00 | ORB-short ORB[179.56,181.99] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:15:00 | 177.71 | 179.24 | 0.00 | T1 1.5R @ 177.71 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 178.49 | 178.44 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 176.58 | 177.13 | 0.00 | ORB-short ORB[176.60,178.60] vol=2.0x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-03-10 11:10:00 | 177.00 | 177.09 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:50:00 | 178.20 | 176.33 | 0.00 | ORB-long ORB[174.31,175.98] vol=2.4x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-03-19 10:55:00 | 177.50 | 176.41 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:25:00 | 181.21 | 179.34 | 0.00 | ORB-long ORB[177.81,178.85] vol=1.8x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:40:00 | 182.04 | 179.98 | 0.00 | T1 1.5R @ 182.04 |
| Target hit | 2026-04-10 14:25:00 | 181.93 | 181.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 183.65 | 182.60 | 0.00 | ORB-long ORB[181.40,183.46] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-04-15 10:45:00 | 183.02 | 183.04 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 182.71 | 183.91 | 0.00 | ORB-short ORB[183.00,185.19] vol=2.1x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:50:00 | 181.93 | 183.48 | 0.00 | T1 1.5R @ 181.93 |
| Target hit | 2026-04-16 15:20:00 | 182.02 | 182.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 186.41 | 185.90 | 0.00 | ORB-long ORB[184.88,185.96] vol=2.1x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:40:00 | 187.33 | 186.25 | 0.00 | T1 1.5R @ 187.33 |
| Target hit | 2026-04-21 11:50:00 | 188.70 | 188.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 200.40 | 199.10 | 0.00 | ORB-long ORB[197.44,200.00] vol=3.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:10:00 | 201.32 | 199.76 | 0.00 | T1 1.5R @ 201.32 |
| Stop hit — per-position SL triggered | 2026-05-04 11:20:00 | 200.40 | 199.87 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 207.05 | 208.39 | 0.00 | ORB-short ORB[208.50,211.50] vol=1.8x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 206.11 | 208.15 | 0.00 | T1 1.5R @ 206.11 |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 207.05 | 207.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:45:00 | 209.99 | 2026-02-09 10:55:00 | 211.28 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2026-02-10 09:45:00 | 211.03 | 2026-02-10 10:55:00 | 210.04 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-10 09:45:00 | 211.03 | 2026-02-10 15:20:00 | 209.82 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-11 10:30:00 | 210.00 | 2026-02-11 11:20:00 | 209.39 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-11 10:30:00 | 210.00 | 2026-02-11 13:40:00 | 210.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 205.30 | 2026-02-17 09:40:00 | 206.17 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 09:30:00 | 205.30 | 2026-02-17 09:50:00 | 205.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:50:00 | 201.95 | 2026-02-18 12:05:00 | 202.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-19 11:10:00 | 201.31 | 2026-02-19 11:30:00 | 201.67 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-03-04 11:05:00 | 185.39 | 2026-03-04 11:25:00 | 184.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-05 09:55:00 | 178.49 | 2026-03-05 10:15:00 | 177.71 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-05 09:55:00 | 178.49 | 2026-03-05 11:15:00 | 178.49 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 11:00:00 | 176.58 | 2026-03-10 11:10:00 | 177.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-19 10:50:00 | 178.20 | 2026-03-19 10:55:00 | 177.50 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-10 10:25:00 | 181.21 | 2026-04-10 10:40:00 | 182.04 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-10 10:25:00 | 181.21 | 2026-04-10 14:25:00 | 181.93 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-15 09:40:00 | 183.65 | 2026-04-15 10:45:00 | 183.02 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 09:35:00 | 182.71 | 2026-04-16 09:50:00 | 181.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-16 09:35:00 | 182.71 | 2026-04-16 15:20:00 | 182.02 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-21 09:35:00 | 186.41 | 2026-04-21 09:40:00 | 187.33 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:35:00 | 186.41 | 2026-04-21 11:50:00 | 188.70 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2026-05-04 11:00:00 | 200.40 | 2026-05-04 11:10:00 | 201.32 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-05-04 11:00:00 | 200.40 | 2026-05-04 11:20:00 | 200.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:50:00 | 207.05 | 2026-05-08 11:30:00 | 206.11 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-05-08 10:50:00 | 207.05 | 2026-05-08 12:15:00 | 207.05 | STOP_HIT | 0.50 | 0.00% |
