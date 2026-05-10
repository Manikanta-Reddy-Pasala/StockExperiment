# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 66.15
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 5
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.06% | 0.6% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.06% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.10% | 0.5% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.10% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 0 | 9 | 5 | 0.08% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 70.07 | 69.69 | 0.00 | ORB-long ORB[69.00,70.00] vol=2.0x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 69.76 | 69.79 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 67.49 | 67.85 | 0.00 | ORB-short ORB[67.71,68.55] vol=2.5x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 67.16 | 67.79 | 0.00 | T1 1.5R @ 67.16 |
| Stop hit — per-position SL triggered | 2026-02-27 13:30:00 | 67.49 | 67.64 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 59.99 | 59.50 | 0.00 | ORB-long ORB[58.70,59.49] vol=4.0x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 59.76 | 59.52 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 65.31 | 64.86 | 0.00 | ORB-long ORB[64.40,65.19] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:30:00 | 65.72 | 65.03 | 0.00 | T1 1.5R @ 65.72 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 65.31 | 65.06 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 65.20 | 64.78 | 0.00 | ORB-long ORB[64.35,64.90] vol=2.4x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:55:00 | 65.52 | 64.90 | 0.00 | T1 1.5R @ 65.52 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 65.20 | 64.90 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 64.00 | 63.20 | 0.00 | ORB-long ORB[62.33,63.01] vol=5.1x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:35:00 | 64.46 | 63.43 | 0.00 | T1 1.5R @ 64.46 |
| Stop hit — per-position SL triggered | 2026-04-27 12:10:00 | 64.00 | 63.81 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 64.86 | 64.58 | 0.00 | ORB-long ORB[64.02,64.80] vol=2.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-05-04 09:40:00 | 64.57 | 64.64 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 63.93 | 64.14 | 0.00 | ORB-short ORB[63.94,64.55] vol=2.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:05:00 | 63.64 | 64.09 | 0.00 | T1 1.5R @ 63.64 |
| Stop hit — per-position SL triggered | 2026-05-06 10:30:00 | 63.93 | 64.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 66.67 | 66.99 | 0.00 | ORB-short ORB[66.70,67.70] vol=2.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-05-08 10:35:00 | 66.97 | 66.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 09:45:00 | 70.07 | 2026-02-20 10:05:00 | 69.76 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-27 10:20:00 | 67.49 | 2026-02-27 10:50:00 | 67.16 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-27 10:20:00 | 67.49 | 2026-02-27 13:30:00 | 67.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 59.99 | 2026-04-10 09:35:00 | 59.76 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-21 10:00:00 | 65.31 | 2026-04-21 10:30:00 | 65.72 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-21 10:00:00 | 65.31 | 2026-04-21 11:00:00 | 65.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 65.20 | 2026-04-22 10:55:00 | 65.52 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 10:45:00 | 65.20 | 2026-04-22 11:00:00 | 65.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:10:00 | 64.00 | 2026-04-27 10:35:00 | 64.46 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-27 10:10:00 | 64.00 | 2026-04-27 12:10:00 | 64.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:30:00 | 64.86 | 2026-05-04 09:40:00 | 64.57 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-05-06 09:55:00 | 63.93 | 2026-05-06 10:05:00 | 63.64 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-06 09:55:00 | 63.93 | 2026-05-06 10:30:00 | 63.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:35:00 | 66.67 | 2026-05-08 10:35:00 | 66.97 | STOP_HIT | 1.00 | -0.45% |
