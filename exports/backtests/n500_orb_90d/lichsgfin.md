# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 581.85
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 4
- **Avg / median % per leg:** -0.03% / -0.26%
- **Sum % (uncompounded):** -0.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 1.9% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.31% | -2.5% |
| SELL @ 2nd Alert (retest1) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.31% | -2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 7 | 33.3% | 3 | 14 | 4 | -0.03% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 509.70 | 511.94 | 0.00 | ORB-short ORB[510.75,515.35] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 511.04 | 511.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 515.20 | 512.42 | 0.00 | ORB-long ORB[509.65,511.85] vol=2.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:20:00 | 516.90 | 513.72 | 0.00 | T1 1.5R @ 516.90 |
| Target hit | 2026-02-17 15:20:00 | 518.50 | 517.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 520.90 | 519.96 | 0.00 | ORB-long ORB[517.55,520.65] vol=1.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 519.84 | 520.19 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 531.50 | 530.13 | 0.00 | ORB-long ORB[525.60,531.30] vol=6.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 529.92 | 530.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 527.50 | 525.06 | 0.00 | ORB-long ORB[517.55,521.25] vol=5.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 525.56 | 526.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 525.45 | 523.59 | 0.00 | ORB-long ORB[521.70,525.05] vol=2.4x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:35:00 | 527.47 | 524.82 | 0.00 | T1 1.5R @ 527.47 |
| Target hit | 2026-02-24 13:30:00 | 527.55 | 527.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 540.10 | 537.18 | 0.00 | ORB-long ORB[531.60,538.00] vol=1.8x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:20:00 | 542.24 | 537.73 | 0.00 | T1 1.5R @ 542.24 |
| Target hit | 2026-02-25 12:45:00 | 543.70 | 543.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 538.65 | 541.64 | 0.00 | ORB-short ORB[541.40,546.25] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 540.42 | 541.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 498.20 | 494.69 | 0.00 | ORB-long ORB[495.00,497.95] vol=1.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:25:00 | 500.76 | 495.57 | 0.00 | T1 1.5R @ 500.76 |
| Stop hit — per-position SL triggered | 2026-03-13 12:35:00 | 498.20 | 496.83 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 494.00 | 494.64 | 0.00 | ORB-short ORB[494.50,499.95] vol=2.3x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-03-19 10:35:00 | 495.74 | 494.61 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 11:05:00 | 527.80 | 524.53 | 0.00 | ORB-long ORB[520.00,527.35] vol=1.9x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-04-08 11:35:00 | 526.40 | 524.87 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 531.10 | 534.16 | 0.00 | ORB-short ORB[532.70,538.45] vol=1.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-04-10 12:45:00 | 532.67 | 532.65 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 539.05 | 537.02 | 0.00 | ORB-long ORB[532.60,538.30] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-04-17 10:35:00 | 537.59 | 537.19 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 559.05 | 561.43 | 0.00 | ORB-short ORB[561.05,568.55] vol=1.8x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-04-22 12:05:00 | 560.68 | 561.00 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 546.70 | 549.37 | 0.00 | ORB-short ORB[548.05,553.75] vol=2.1x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 548.67 | 549.21 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 551.45 | 552.88 | 0.00 | ORB-short ORB[552.05,557.60] vol=1.9x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 553.15 | 553.25 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:35:00 | 581.00 | 583.52 | 0.00 | ORB-short ORB[583.00,587.50] vol=2.2x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-05-08 10:40:00 | 582.64 | 583.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:35:00 | 509.70 | 2026-02-13 09:40:00 | 511.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-17 10:25:00 | 515.20 | 2026-02-17 11:20:00 | 516.90 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-17 10:25:00 | 515.20 | 2026-02-17 15:20:00 | 518.50 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-18 09:35:00 | 520.90 | 2026-02-18 09:50:00 | 519.84 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-19 09:30:00 | 531.50 | 2026-02-19 09:50:00 | 529.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-20 09:45:00 | 527.50 | 2026-02-20 12:55:00 | 525.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-24 09:35:00 | 525.45 | 2026-02-24 10:35:00 | 527.47 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-24 09:35:00 | 525.45 | 2026-02-24 13:30:00 | 527.55 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-25 10:15:00 | 540.10 | 2026-02-25 10:20:00 | 542.24 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-25 10:15:00 | 540.10 | 2026-02-25 12:45:00 | 543.70 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-27 10:20:00 | 538.65 | 2026-02-27 10:35:00 | 540.42 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-13 10:50:00 | 498.20 | 2026-03-13 11:25:00 | 500.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-13 10:50:00 | 498.20 | 2026-03-13 12:35:00 | 498.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:20:00 | 494.00 | 2026-03-19 10:35:00 | 495.74 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-08 11:05:00 | 527.80 | 2026-04-08 11:35:00 | 526.40 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-10 10:05:00 | 531.10 | 2026-04-10 12:45:00 | 532.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 10:30:00 | 539.05 | 2026-04-17 10:35:00 | 537.59 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-22 10:45:00 | 559.05 | 2026-04-22 12:05:00 | 560.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-29 09:35:00 | 546.70 | 2026-04-29 09:45:00 | 548.67 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-05 09:35:00 | 551.45 | 2026-05-05 09:40:00 | 553.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-08 10:35:00 | 581.00 | 2026-05-08 10:40:00 | 582.64 | STOP_HIT | 1.00 | -0.28% |
