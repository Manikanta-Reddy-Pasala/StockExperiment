# Go Digit General Insurance Ltd. (GODIGIT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 313.05
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
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 17
- **Target hits / Stop hits / Partials:** 0 / 17 / 5
- **Avg / median % per leg:** -0.04% / -0.19%
- **Sum % (uncompounded):** -0.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.14% | 1.8% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.14% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -0.29% | -2.6% |
| SELL @ 2nd Alert (retest1) | 9 | 0 | 0.0% | 0 | 9 | 0 | -0.29% | -2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 5 | 22.7% | 0 | 17 | 5 | -0.04% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 313.75 | 311.51 | 0.00 | ORB-long ORB[308.55,312.35] vol=4.3x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 315.62 | 312.19 | 0.00 | T1 1.5R @ 315.62 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 313.75 | 312.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:05:00 | 313.85 | 315.15 | 0.00 | ORB-short ORB[314.40,317.15] vol=1.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-02-13 11:10:00 | 314.78 | 314.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 334.45 | 333.06 | 0.00 | ORB-long ORB[330.05,332.05] vol=3.2x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-20 12:10:00 | 333.65 | 333.35 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 333.00 | 334.63 | 0.00 | ORB-short ORB[333.35,337.15] vol=4.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-23 09:55:00 | 334.19 | 334.41 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 338.40 | 339.51 | 0.00 | ORB-short ORB[338.65,343.45] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-27 10:10:00 | 339.28 | 339.17 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 329.75 | 331.86 | 0.00 | ORB-short ORB[330.25,334.65] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-03-16 11:35:00 | 330.96 | 331.44 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 332.15 | 334.95 | 0.00 | ORB-short ORB[334.25,338.15] vol=2.2x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 333.37 | 334.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 329.70 | 331.84 | 0.00 | ORB-short ORB[330.75,334.30] vol=1.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-03-18 11:30:00 | 330.64 | 331.25 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 330.45 | 328.47 | 0.00 | ORB-long ORB[325.00,328.40] vol=3.9x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:55:00 | 331.88 | 329.59 | 0.00 | T1 1.5R @ 331.88 |
| Stop hit — per-position SL triggered | 2026-03-19 10:00:00 | 330.45 | 329.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 10:40:00 | 326.60 | 325.12 | 0.00 | ORB-long ORB[319.10,323.55] vol=2.2x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:10:00 | 328.86 | 325.63 | 0.00 | T1 1.5R @ 328.86 |
| Stop hit — per-position SL triggered | 2026-03-30 11:25:00 | 326.60 | 326.64 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 319.95 | 321.08 | 0.00 | ORB-short ORB[320.05,324.00] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-04-15 11:20:00 | 320.57 | 320.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 325.85 | 322.60 | 0.00 | ORB-long ORB[320.35,323.60] vol=10.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:30:00 | 327.43 | 324.68 | 0.00 | T1 1.5R @ 327.43 |
| Stop hit — per-position SL triggered | 2026-04-22 11:35:00 | 325.85 | 324.71 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 323.40 | 321.92 | 0.00 | ORB-long ORB[319.70,322.80] vol=3.1x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-04-23 10:20:00 | 322.47 | 321.96 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 322.55 | 320.31 | 0.00 | ORB-long ORB[319.00,321.50] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 323.69 | 320.57 | 0.00 | T1 1.5R @ 323.69 |
| Stop hit — per-position SL triggered | 2026-04-28 11:30:00 | 322.55 | 320.78 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 304.00 | 306.06 | 0.00 | ORB-short ORB[308.15,312.15] vol=11.9x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 304.80 | 305.86 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 302.10 | 303.59 | 0.00 | ORB-short ORB[303.50,305.80] vol=2.3x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-05-05 14:20:00 | 302.85 | 303.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 307.15 | 305.38 | 0.00 | ORB-long ORB[304.25,306.00] vol=4.7x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-05-07 10:50:00 | 306.47 | 305.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 313.75 | 2026-02-09 11:20:00 | 315.62 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-09 11:00:00 | 313.75 | 2026-02-09 11:30:00 | 313.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:05:00 | 313.85 | 2026-02-13 11:10:00 | 314.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-20 11:05:00 | 334.45 | 2026-02-20 12:10:00 | 333.65 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-23 09:45:00 | 333.00 | 2026-02-23 09:55:00 | 334.19 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-27 09:40:00 | 338.40 | 2026-02-27 10:10:00 | 339.28 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-16 10:35:00 | 329.75 | 2026-03-16 11:35:00 | 330.96 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-17 11:05:00 | 332.15 | 2026-03-17 11:25:00 | 333.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-18 10:40:00 | 329.70 | 2026-03-18 11:30:00 | 330.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-19 09:50:00 | 330.45 | 2026-03-19 09:55:00 | 331.88 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-19 09:50:00 | 330.45 | 2026-03-19 10:00:00 | 330.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-30 10:40:00 | 326.60 | 2026-03-30 11:10:00 | 328.86 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-03-30 10:40:00 | 326.60 | 2026-03-30 11:25:00 | 326.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 11:00:00 | 319.95 | 2026-04-15 11:20:00 | 320.57 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-22 11:05:00 | 325.85 | 2026-04-22 11:30:00 | 327.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-22 11:05:00 | 325.85 | 2026-04-22 11:35:00 | 325.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:00:00 | 323.40 | 2026-04-23 10:20:00 | 322.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-28 11:10:00 | 322.55 | 2026-04-28 11:20:00 | 323.69 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-28 11:10:00 | 322.55 | 2026-04-28 11:30:00 | 322.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 11:00:00 | 304.00 | 2026-05-04 11:30:00 | 304.80 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-05 11:00:00 | 302.10 | 2026-05-05 14:20:00 | 302.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-07 10:45:00 | 307.15 | 2026-05-07 10:50:00 | 306.47 | STOP_HIT | 1.00 | -0.22% |
