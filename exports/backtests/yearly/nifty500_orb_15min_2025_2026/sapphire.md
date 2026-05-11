# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2025-12-08 09:15:00 → 2026-05-08 15:25:00 (7650 bars)
- **Last close:** 183.60
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 12
- **Target hits / Stop hits / Partials:** 5 / 12 / 12
- **Avg / median % per leg:** 0.50% / 0.34%
- **Sum % (uncompounded):** 14.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 8 | 53.3% | 1 | 7 | 7 | 0.29% | 4.4% |
| BUY @ 2nd Alert (retest1) | 15 | 8 | 53.3% | 1 | 7 | 7 | 0.29% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.72% | 10.1% |
| SELL @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.72% | 10.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 17 | 58.6% | 5 | 12 | 12 | 0.50% | 14.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:45:00 | 239.79 | 240.68 | 0.00 | ORB-short ORB[241.49,244.75] vol=1.5x ATR=1.33 |
| Target hit | 2025-12-08 15:20:00 | 238.99 | 239.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:55:00 | 238.46 | 237.99 | 0.00 | ORB-long ORB[236.52,238.40] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 11:20:00 | 239.29 | 238.10 | 0.00 | T1 1.5R @ 239.29 |
| Stop hit — per-position SL triggered | 2025-12-12 11:55:00 | 238.46 | 238.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 234.00 | 232.56 | 0.00 | ORB-long ORB[228.66,232.02] vol=8.8x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:50:00 | 235.94 | 232.73 | 0.00 | T1 1.5R @ 235.94 |
| Stop hit — per-position SL triggered | 2025-12-22 11:10:00 | 234.00 | 233.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 253.01 | 251.26 | 0.00 | ORB-long ORB[248.37,252.12] vol=2.0x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:35:00 | 253.88 | 251.56 | 0.00 | T1 1.5R @ 253.88 |
| Stop hit — per-position SL triggered | 2025-12-26 11:45:00 | 253.01 | 251.76 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:45:00 | 248.72 | 249.21 | 0.00 | ORB-short ORB[248.76,251.01] vol=1.7x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-12-30 09:50:00 | 249.49 | 249.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:00:00 | 222.60 | 220.40 | 0.00 | ORB-long ORB[218.05,221.30] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:30:00 | 224.07 | 222.37 | 0.00 | T1 1.5R @ 224.07 |
| Stop hit — per-position SL triggered | 2026-01-09 13:35:00 | 222.60 | 222.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-01-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:30:00 | 218.65 | 220.15 | 0.00 | ORB-short ORB[219.00,221.90] vol=2.1x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:35:00 | 217.25 | 220.00 | 0.00 | T1 1.5R @ 217.25 |
| Stop hit — per-position SL triggered | 2026-01-12 12:05:00 | 218.65 | 219.91 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:50:00 | 222.90 | 224.10 | 0.00 | ORB-short ORB[223.95,226.40] vol=1.9x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:10:00 | 221.94 | 223.87 | 0.00 | T1 1.5R @ 221.94 |
| Stop hit — per-position SL triggered | 2026-01-16 13:30:00 | 222.90 | 222.97 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-01-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:10:00 | 218.60 | 219.94 | 0.00 | ORB-short ORB[219.30,222.20] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 12:10:00 | 217.81 | 219.35 | 0.00 | T1 1.5R @ 217.81 |
| Target hit | 2026-01-19 15:20:00 | 215.45 | 217.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:15:00 | 214.45 | 215.87 | 0.00 | ORB-short ORB[215.30,217.60] vol=4.0x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:30:00 | 212.85 | 215.13 | 0.00 | T1 1.5R @ 212.85 |
| Target hit | 2026-01-20 15:20:00 | 204.80 | 211.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-01-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:50:00 | 198.00 | 201.06 | 0.00 | ORB-short ORB[200.60,202.35] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:40:00 | 196.59 | 199.25 | 0.00 | T1 1.5R @ 196.59 |
| Target hit | 2026-01-22 15:20:00 | 193.60 | 196.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-01-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 09:40:00 | 184.60 | 183.35 | 0.00 | ORB-long ORB[181.90,184.45] vol=4.3x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:45:00 | 186.06 | 184.75 | 0.00 | T1 1.5R @ 186.06 |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 184.60 | 184.98 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 09:30:00 | 188.77 | 188.18 | 0.00 | ORB-long ORB[185.99,188.45] vol=2.4x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 09:35:00 | 190.36 | 189.08 | 0.00 | T1 1.5R @ 190.36 |
| Target hit | 2026-02-01 10:15:00 | 189.10 | 189.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:15:00 | 152.68 | 155.37 | 0.00 | ORB-short ORB[156.03,157.85] vol=1.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-03-27 10:30:00 | 153.53 | 154.40 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 179.80 | 177.58 | 0.00 | ORB-long ORB[172.91,175.00] vol=2.4x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 178.64 | 177.88 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 180.65 | 176.27 | 0.00 | ORB-long ORB[173.38,175.89] vol=4.0x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 182.55 | 178.13 | 0.00 | T1 1.5R @ 182.55 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 180.65 | 179.56 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 184.43 | 184.60 | 0.00 | ORB-short ORB[185.20,187.05] vol=3.0x ATR=0.72 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 185.15 | 184.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-12-08 10:45:00 | 239.79 | 2025-12-08 15:20:00 | 238.99 | TARGET_HIT | 1.00 | 0.33% |
| BUY | retest1 | 2025-12-12 10:55:00 | 238.46 | 2025-12-12 11:20:00 | 239.29 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-12 10:55:00 | 238.46 | 2025-12-12 11:55:00 | 238.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 10:40:00 | 234.00 | 2025-12-22 10:50:00 | 235.94 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-12-22 10:40:00 | 234.00 | 2025-12-22 11:10:00 | 234.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 11:10:00 | 253.01 | 2025-12-26 11:35:00 | 253.88 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-26 11:10:00 | 253.01 | 2025-12-26 11:45:00 | 253.01 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 09:45:00 | 248.72 | 2025-12-30 09:50:00 | 249.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-09 10:00:00 | 222.60 | 2026-01-09 13:30:00 | 224.07 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-01-09 10:00:00 | 222.60 | 2026-01-09 13:35:00 | 222.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-12 10:30:00 | 218.65 | 2026-01-12 11:35:00 | 217.25 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-01-12 10:30:00 | 218.65 | 2026-01-12 12:05:00 | 218.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-16 10:50:00 | 222.90 | 2026-01-16 11:10:00 | 221.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-16 10:50:00 | 222.90 | 2026-01-16 13:30:00 | 222.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 11:10:00 | 218.60 | 2026-01-19 12:10:00 | 217.81 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-19 11:10:00 | 218.60 | 2026-01-19 15:20:00 | 215.45 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2026-01-20 10:15:00 | 214.45 | 2026-01-20 11:30:00 | 212.85 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-01-20 10:15:00 | 214.45 | 2026-01-20 15:20:00 | 204.80 | TARGET_HIT | 0.50 | 4.50% |
| SELL | retest1 | 2026-01-22 10:50:00 | 198.00 | 2026-01-22 11:40:00 | 196.59 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-01-22 10:50:00 | 198.00 | 2026-01-22 15:20:00 | 193.60 | TARGET_HIT | 0.50 | 2.22% |
| BUY | retest1 | 2026-01-28 09:40:00 | 184.60 | 2026-01-28 09:45:00 | 186.06 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-01-28 09:40:00 | 184.60 | 2026-01-28 10:15:00 | 184.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 09:30:00 | 188.77 | 2026-02-01 09:35:00 | 190.36 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2026-02-01 09:30:00 | 188.77 | 2026-02-01 10:15:00 | 189.10 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-03-27 10:15:00 | 152.68 | 2026-03-27 10:30:00 | 153.53 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-04-21 10:55:00 | 179.80 | 2026-04-21 11:15:00 | 178.64 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2026-04-29 10:10:00 | 180.65 | 2026-04-29 10:15:00 | 182.55 | PARTIAL | 0.50 | 1.05% |
| BUY | retest1 | 2026-04-29 10:10:00 | 180.65 | 2026-04-29 10:40:00 | 180.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:45:00 | 184.43 | 2026-05-08 11:00:00 | 185.15 | STOP_HIT | 1.00 | -0.39% |
