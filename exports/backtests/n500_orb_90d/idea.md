# Vodafone Idea Ltd. (IDEA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 11.24
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 6
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 5.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.33% | 3.3% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.33% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.17% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 8 | 40.0% | 2 | 12 | 6 | 0.25% | 5.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 11.38 | 11.44 | 0.00 | ORB-short ORB[11.42,11.51] vol=2.6x ATR=0.04 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 11.42 | 11.43 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 11.40 | 11.34 | 0.00 | ORB-long ORB[11.21,11.38] vol=1.6x ATR=0.04 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 11.36 | 11.34 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 11.03 | 11.11 | 0.00 | ORB-short ORB[11.07,11.22] vol=1.9x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 11.08 | 11.11 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 10.89 | 10.93 | 0.00 | ORB-short ORB[10.96,11.05] vol=4.5x ATR=0.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 10.84 | 10.92 | 0.00 | T1 1.5R @ 10.84 |
| Stop hit — per-position SL triggered | 2026-02-24 14:10:00 | 10.89 | 10.89 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 9.46 | 9.51 | 0.00 | ORB-short ORB[9.47,9.59] vol=1.7x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:50:00 | 9.40 | 9.48 | 0.00 | T1 1.5R @ 9.40 |
| Target hit | 2026-03-13 15:20:00 | 9.31 | 9.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 9.27 | 9.31 | 0.00 | ORB-short ORB[9.28,9.38] vol=2.4x ATR=0.04 |
| Stop hit — per-position SL triggered | 2026-03-18 10:50:00 | 9.31 | 9.30 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:10:00 | 9.42 | 9.23 | 0.00 | ORB-long ORB[9.05,9.19] vol=1.8x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 9.37 | 9.23 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 8.84 | 8.79 | 0.00 | ORB-long ORB[8.73,8.83] vol=1.8x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-03-30 09:45:00 | 8.79 | 8.79 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 9.01 | 8.96 | 0.00 | ORB-long ORB[8.90,9.00] vol=1.8x ATR=0.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:50:00 | 9.09 | 9.00 | 0.00 | T1 1.5R @ 9.09 |
| Target hit | 2026-04-08 15:20:00 | 9.23 | 9.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 9.63 | 9.55 | 0.00 | ORB-long ORB[9.42,9.54] vol=2.1x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 9.69 | 9.59 | 0.00 | T1 1.5R @ 9.69 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 9.63 | 9.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 9.61 | 9.53 | 0.00 | ORB-long ORB[9.48,9.58] vol=2.3x ATR=0.03 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 9.58 | 9.53 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 9.50 | 9.55 | 0.00 | ORB-short ORB[9.51,9.61] vol=1.7x ATR=0.03 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 9.53 | 9.54 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 9.45 | 9.53 | 0.00 | ORB-short ORB[9.52,9.63] vol=3.2x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 9.40 | 9.49 | 0.00 | T1 1.5R @ 9.40 |
| Stop hit — per-position SL triggered | 2026-04-24 10:10:00 | 9.45 | 9.48 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 10.96 | 10.72 | 0.00 | ORB-long ORB[10.50,10.61] vol=4.9x ATR=0.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:45:00 | 11.07 | 10.90 | 0.00 | T1 1.5R @ 11.07 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 10.96 | 10.94 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 11.38 | 2026-02-13 09:40:00 | 11.42 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-16 09:30:00 | 11.40 | 2026-02-16 09:35:00 | 11.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-23 09:40:00 | 11.03 | 2026-02-23 09:50:00 | 11.08 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-24 10:55:00 | 10.89 | 2026-02-24 11:45:00 | 10.84 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-24 10:55:00 | 10.89 | 2026-02-24 14:10:00 | 10.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:40:00 | 9.46 | 2026-03-13 09:50:00 | 9.40 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-13 09:40:00 | 9.46 | 2026-03-13 15:20:00 | 9.31 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2026-03-18 09:35:00 | 9.27 | 2026-03-18 10:50:00 | 9.31 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-20 11:10:00 | 9.42 | 2026-03-20 11:15:00 | 9.37 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-03-30 09:30:00 | 8.84 | 2026-03-30 09:45:00 | 8.79 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-08 09:35:00 | 9.01 | 2026-04-08 10:50:00 | 9.09 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-08 09:35:00 | 9.01 | 2026-04-08 15:20:00 | 9.23 | TARGET_HIT | 0.50 | 2.44% |
| BUY | retest1 | 2026-04-21 09:50:00 | 9.63 | 2026-04-21 10:10:00 | 9.69 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-21 09:50:00 | 9.63 | 2026-04-21 11:30:00 | 9.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 9.61 | 2026-04-22 10:55:00 | 9.58 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-23 09:30:00 | 9.50 | 2026-04-23 09:50:00 | 9.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 09:30:00 | 9.45 | 2026-04-24 09:50:00 | 9.40 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-24 09:30:00 | 9.45 | 2026-04-24 10:10:00 | 9.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 10.96 | 2026-05-05 09:45:00 | 11.07 | PARTIAL | 0.50 | 1.04% |
| BUY | retest1 | 2026-05-05 09:40:00 | 10.96 | 2026-05-05 10:05:00 | 10.96 | STOP_HIT | 0.50 | 0.00% |
