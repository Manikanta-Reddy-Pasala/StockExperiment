# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 185.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 5
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 3.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.29% | 3.4% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.29% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.09% | 0.5% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.09% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 3 | 10 | 5 | 0.22% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 155.35 | 154.89 | 0.00 | ORB-long ORB[154.41,155.29] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 155.00 | 154.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 153.87 | 154.23 | 0.00 | ORB-short ORB[154.00,155.23] vol=2.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 153.35 | 153.99 | 0.00 | T1 1.5R @ 153.35 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 153.87 | 153.81 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 152.30 | 151.53 | 0.00 | ORB-long ORB[150.33,151.95] vol=1.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-16 09:45:00 | 151.79 | 151.67 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 153.50 | 152.17 | 0.00 | ORB-long ORB[150.82,152.61] vol=1.6x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:55:00 | 154.51 | 153.26 | 0.00 | T1 1.5R @ 154.51 |
| Target hit | 2026-02-20 12:10:00 | 154.56 | 154.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 161.20 | 160.98 | 0.00 | ORB-long ORB[159.79,161.18] vol=1.7x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 160.78 | 160.94 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:10:00 | 152.00 | 153.12 | 0.00 | ORB-short ORB[152.98,154.90] vol=1.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 152.49 | 153.05 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 155.99 | 157.28 | 0.00 | ORB-short ORB[156.18,158.50] vol=2.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 156.58 | 156.94 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 158.40 | 157.93 | 0.00 | ORB-long ORB[156.75,158.38] vol=2.0x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 157.91 | 158.06 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 157.14 | 156.04 | 0.00 | ORB-long ORB[155.00,156.74] vol=2.2x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:50:00 | 158.02 | 156.65 | 0.00 | T1 1.5R @ 158.02 |
| Target hit | 2026-04-13 14:05:00 | 161.42 | 161.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:30:00 | 164.40 | 163.28 | 0.00 | ORB-long ORB[162.30,163.80] vol=2.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 163.79 | 163.46 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 165.40 | 164.32 | 0.00 | ORB-long ORB[163.82,165.00] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 164.85 | 164.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 190.50 | 189.45 | 0.00 | ORB-long ORB[187.78,190.20] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:45:00 | 191.71 | 189.78 | 0.00 | T1 1.5R @ 191.71 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 190.50 | 189.80 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 185.50 | 185.67 | 0.00 | ORB-short ORB[185.61,187.45] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:45:00 | 184.69 | 185.62 | 0.00 | T1 1.5R @ 184.69 |
| Target hit | 2026-05-08 15:20:00 | 184.65 | 185.24 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:25:00 | 155.35 | 2026-02-10 10:40:00 | 155.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-11 09:30:00 | 153.87 | 2026-02-11 09:40:00 | 153.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-11 09:30:00 | 153.87 | 2026-02-11 10:15:00 | 153.87 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:30:00 | 152.30 | 2026-02-16 09:45:00 | 151.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 09:40:00 | 153.50 | 2026-02-20 09:55:00 | 154.51 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-02-20 09:40:00 | 153.50 | 2026-02-20 12:10:00 | 154.56 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-26 09:50:00 | 161.20 | 2026-02-26 09:55:00 | 160.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-01 11:10:00 | 152.00 | 2026-04-01 11:25:00 | 152.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-09 09:50:00 | 155.99 | 2026-04-09 10:15:00 | 156.58 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:40:00 | 158.40 | 2026-04-10 09:55:00 | 157.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-13 09:40:00 | 157.14 | 2026-04-13 09:50:00 | 158.02 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-13 09:40:00 | 157.14 | 2026-04-13 14:05:00 | 161.42 | TARGET_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2026-04-15 10:30:00 | 164.40 | 2026-04-15 10:50:00 | 163.79 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-16 10:25:00 | 165.40 | 2026-04-16 10:40:00 | 164.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-28 09:40:00 | 190.50 | 2026-04-28 09:45:00 | 191.71 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 09:40:00 | 190.50 | 2026-04-28 09:50:00 | 190.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:15:00 | 185.50 | 2026-05-08 12:45:00 | 184.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-08 11:15:00 | 185.50 | 2026-05-08 15:20:00 | 184.65 | TARGET_HIT | 0.50 | 0.46% |
