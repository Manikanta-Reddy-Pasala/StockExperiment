# JSW Infrastructure Ltd. (JSWINFRA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 284.50
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 1 / 17 / 7
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.11% | 1.9% |
| BUY @ 2nd Alert (retest1) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.11% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.05% | -0.4% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.05% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 8 | 32.0% | 1 | 17 | 7 | 0.06% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 262.35 | 261.08 | 0.00 | ORB-long ORB[259.60,262.00] vol=2.0x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:40:00 | 263.82 | 261.55 | 0.00 | T1 1.5R @ 263.82 |
| Target hit | 2026-02-09 15:20:00 | 266.40 | 266.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 262.10 | 263.00 | 0.00 | ORB-short ORB[263.15,265.90] vol=3.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:40:00 | 261.25 | 262.77 | 0.00 | T1 1.5R @ 261.25 |
| Stop hit — per-position SL triggered | 2026-02-11 13:30:00 | 262.10 | 262.27 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:30:00 | 262.35 | 261.18 | 0.00 | ORB-long ORB[260.35,261.75] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-02-12 10:35:00 | 261.77 | 261.21 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 258.75 | 259.79 | 0.00 | ORB-short ORB[259.20,261.00] vol=2.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 259.44 | 259.54 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 254.20 | 255.82 | 0.00 | ORB-short ORB[254.25,257.55] vol=2.2x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 255.03 | 255.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 257.70 | 255.22 | 0.00 | ORB-long ORB[253.50,255.95] vol=2.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:25:00 | 259.08 | 256.19 | 0.00 | T1 1.5R @ 259.08 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 257.70 | 256.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 266.00 | 264.94 | 0.00 | ORB-long ORB[262.90,265.95] vol=1.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 265.07 | 265.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 258.45 | 255.97 | 0.00 | ORB-long ORB[254.30,257.25] vol=1.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-03-12 10:00:00 | 257.36 | 256.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 256.60 | 254.15 | 0.00 | ORB-long ORB[252.35,255.40] vol=2.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 258.63 | 255.51 | 0.00 | T1 1.5R @ 258.63 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 256.60 | 255.52 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 256.50 | 255.82 | 0.00 | ORB-long ORB[253.50,256.35] vol=3.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:50:00 | 257.87 | 256.14 | 0.00 | T1 1.5R @ 257.87 |
| Stop hit — per-position SL triggered | 2026-03-18 10:00:00 | 256.50 | 256.29 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 270.97 | 269.59 | 0.00 | ORB-long ORB[267.48,269.69] vol=3.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 269.76 | 270.00 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 270.39 | 272.28 | 0.00 | ORB-short ORB[272.20,275.00] vol=1.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 271.33 | 272.25 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 275.10 | 273.86 | 0.00 | ORB-long ORB[272.14,274.59] vol=1.8x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-04-17 09:35:00 | 274.14 | 273.99 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 280.74 | 279.50 | 0.00 | ORB-long ORB[276.50,280.19] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 279.29 | 279.78 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 273.79 | 276.09 | 0.00 | ORB-short ORB[274.90,278.30] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-24 09:45:00 | 274.89 | 275.75 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:05:00 | 266.64 | 269.08 | 0.00 | ORB-short ORB[268.05,272.01] vol=1.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:50:00 | 265.09 | 267.73 | 0.00 | T1 1.5R @ 265.09 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 266.64 | 267.32 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 275.35 | 273.29 | 0.00 | ORB-long ORB[272.30,275.00] vol=2.9x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 274.45 | 273.71 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 282.70 | 281.00 | 0.00 | ORB-long ORB[279.50,282.50] vol=1.5x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:05:00 | 284.27 | 282.17 | 0.00 | T1 1.5R @ 284.27 |
| Stop hit — per-position SL triggered | 2026-05-08 12:35:00 | 282.70 | 282.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 262.35 | 2026-02-09 11:40:00 | 263.82 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-09 11:05:00 | 262.35 | 2026-02-09 15:20:00 | 266.40 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2026-02-11 11:05:00 | 262.10 | 2026-02-11 11:40:00 | 261.25 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-11 11:05:00 | 262.10 | 2026-02-11 13:30:00 | 262.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:30:00 | 262.35 | 2026-02-12 10:35:00 | 261.77 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 09:30:00 | 258.75 | 2026-02-13 09:40:00 | 259.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-23 10:45:00 | 254.20 | 2026-02-23 11:10:00 | 255.03 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:15:00 | 257.70 | 2026-02-26 10:25:00 | 259.08 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-26 10:15:00 | 257.70 | 2026-02-26 10:30:00 | 257.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:30:00 | 266.00 | 2026-03-11 09:35:00 | 265.07 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-12 09:45:00 | 258.45 | 2026-03-12 10:00:00 | 257.36 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-17 09:45:00 | 256.60 | 2026-03-17 10:20:00 | 258.63 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-03-17 09:45:00 | 256.60 | 2026-03-17 10:25:00 | 256.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:35:00 | 256.50 | 2026-03-18 09:50:00 | 257.87 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-18 09:35:00 | 256.50 | 2026-03-18 10:00:00 | 256.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:45:00 | 270.97 | 2026-04-15 10:25:00 | 269.76 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-16 11:10:00 | 270.39 | 2026-04-16 11:15:00 | 271.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-17 09:30:00 | 275.10 | 2026-04-17 09:35:00 | 274.14 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-21 10:00:00 | 280.74 | 2026-04-21 11:30:00 | 279.29 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-24 09:35:00 | 273.79 | 2026-04-24 09:45:00 | 274.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-30 10:05:00 | 266.64 | 2026-04-30 10:50:00 | 265.09 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-30 10:05:00 | 266.64 | 2026-04-30 11:10:00 | 266.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:45:00 | 275.35 | 2026-05-06 09:55:00 | 274.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-08 10:30:00 | 282.70 | 2026-05-08 12:05:00 | 284.27 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-08 10:30:00 | 282.70 | 2026-05-08 12:35:00 | 282.70 | STOP_HIT | 0.50 | 0.00% |
