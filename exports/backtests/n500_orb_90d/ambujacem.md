# Ambuja Cements Ltd. (AMBUJACEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 443.90
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
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 13
- **Target hits / Stop hits / Partials:** 4 / 13 / 8
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.12% | 1.1% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.12% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.17% | 2.7% |
| SELL @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.17% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.15% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 536.70 | 533.60 | 0.00 | ORB-long ORB[530.35,534.50] vol=1.6x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:05:00 | 539.82 | 535.20 | 0.00 | T1 1.5R @ 539.82 |
| Target hit | 2026-02-09 15:20:00 | 541.20 | 539.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:40:00 | 536.05 | 541.51 | 0.00 | ORB-short ORB[540.45,547.00] vol=3.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 538.00 | 541.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 517.90 | 522.29 | 0.00 | ORB-short ORB[523.25,530.50] vol=1.8x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 519.57 | 521.94 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 524.20 | 522.88 | 0.00 | ORB-long ORB[521.00,523.70] vol=2.0x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:35:00 | 525.75 | 523.46 | 0.00 | T1 1.5R @ 525.75 |
| Target hit | 2026-02-17 11:05:00 | 525.45 | 525.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 514.00 | 515.42 | 0.00 | ORB-short ORB[514.75,516.90] vol=1.6x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:00:00 | 512.16 | 515.24 | 0.00 | T1 1.5R @ 512.16 |
| Target hit | 2026-02-25 15:20:00 | 511.35 | 513.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 502.70 | 505.84 | 0.00 | ORB-short ORB[507.05,512.05] vol=2.1x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:30:00 | 500.93 | 504.23 | 0.00 | T1 1.5R @ 500.93 |
| Stop hit — per-position SL triggered | 2026-02-27 13:00:00 | 502.70 | 503.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 473.95 | 476.37 | 0.00 | ORB-short ORB[477.20,480.30] vol=3.7x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:30:00 | 471.99 | 475.15 | 0.00 | T1 1.5R @ 471.99 |
| Stop hit — per-position SL triggered | 2026-03-05 13:30:00 | 473.95 | 473.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 475.15 | 478.30 | 0.00 | ORB-short ORB[475.70,480.50] vol=1.7x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:40:00 | 472.93 | 476.78 | 0.00 | T1 1.5R @ 472.93 |
| Target hit | 2026-03-06 15:20:00 | 466.20 | 472.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 436.60 | 440.66 | 0.00 | ORB-short ORB[442.10,446.30] vol=1.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 438.08 | 439.88 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 399.75 | 403.09 | 0.00 | ORB-short ORB[401.95,407.90] vol=1.8x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:35:00 | 396.26 | 400.73 | 0.00 | T1 1.5R @ 396.26 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 399.75 | 400.22 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:10:00 | 410.30 | 413.33 | 0.00 | ORB-short ORB[412.80,418.60] vol=3.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2026-03-27 10:50:00 | 412.13 | 411.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:55:00 | 419.90 | 422.95 | 0.00 | ORB-short ORB[420.80,426.95] vol=1.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-04-07 13:35:00 | 421.61 | 421.77 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:40:00 | 457.30 | 450.03 | 0.00 | ORB-long ORB[435.00,441.85] vol=1.6x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-04-08 10:45:00 | 454.38 | 450.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:35:00 | 459.00 | 457.88 | 0.00 | ORB-long ORB[453.45,458.25] vol=2.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 456.99 | 458.36 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 441.85 | 445.07 | 0.00 | ORB-short ORB[444.60,450.20] vol=1.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-04-30 10:45:00 | 443.31 | 444.17 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 440.95 | 438.66 | 0.00 | ORB-long ORB[435.80,439.50] vol=2.5x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:50:00 | 443.13 | 440.19 | 0.00 | T1 1.5R @ 443.13 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 440.95 | 440.56 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 455.20 | 454.53 | 0.00 | ORB-long ORB[446.50,453.00] vol=1.6x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 453.88 | 454.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 536.70 | 2026-02-09 11:05:00 | 539.82 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-09 10:35:00 | 536.70 | 2026-02-09 15:20:00 | 541.20 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2026-02-10 10:40:00 | 536.05 | 2026-02-10 10:50:00 | 538.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-13 10:15:00 | 517.90 | 2026-02-13 10:25:00 | 519.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-17 09:30:00 | 524.20 | 2026-02-17 09:35:00 | 525.75 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-17 09:30:00 | 524.20 | 2026-02-17 11:05:00 | 525.45 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-25 11:00:00 | 514.00 | 2026-02-25 12:00:00 | 512.16 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-25 11:00:00 | 514.00 | 2026-02-25 15:20:00 | 511.35 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-27 11:05:00 | 502.70 | 2026-02-27 11:30:00 | 500.93 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 11:05:00 | 502.70 | 2026-02-27 13:00:00 | 502.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:40:00 | 473.95 | 2026-03-05 11:30:00 | 471.99 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-05 10:40:00 | 473.95 | 2026-03-05 13:30:00 | 473.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 475.15 | 2026-03-06 12:40:00 | 472.93 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-06 10:45:00 | 475.15 | 2026-03-06 15:20:00 | 466.20 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2026-03-13 10:25:00 | 436.60 | 2026-03-13 10:50:00 | 438.08 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-24 09:45:00 | 399.75 | 2026-03-24 10:35:00 | 396.26 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2026-03-24 09:45:00 | 399.75 | 2026-03-24 11:40:00 | 399.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:10:00 | 410.30 | 2026-03-27 10:50:00 | 412.13 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-07 10:55:00 | 419.90 | 2026-04-07 13:35:00 | 421.61 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-08 10:40:00 | 457.30 | 2026-04-08 10:45:00 | 454.38 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2026-04-27 09:35:00 | 459.00 | 2026-04-27 10:10:00 | 456.99 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-30 10:15:00 | 441.85 | 2026-04-30 10:45:00 | 443.31 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-06 09:45:00 | 440.95 | 2026-05-06 09:50:00 | 443.13 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 09:45:00 | 440.95 | 2026-05-06 10:05:00 | 440.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 10:50:00 | 455.20 | 2026-05-08 10:55:00 | 453.88 | STOP_HIT | 1.00 | -0.29% |
