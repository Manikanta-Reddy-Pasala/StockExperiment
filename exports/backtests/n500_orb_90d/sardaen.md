# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 591.90
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
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 12 | 3 | -0.08% | -1.3% |
| BUY @ 2nd Alert (retest1) | 15 | 3 | 20.0% | 0 | 12 | 3 | -0.08% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.17% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 8 | 32.0% | 1 | 17 | 7 | 0.02% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 521.55 | 517.66 | 0.00 | ORB-long ORB[512.30,519.70] vol=2.1x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-02-10 11:35:00 | 519.51 | 517.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 497.40 | 493.99 | 0.00 | ORB-long ORB[491.40,497.25] vol=2.1x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 12:25:00 | 500.46 | 494.94 | 0.00 | T1 1.5R @ 500.46 |
| Stop hit — per-position SL triggered | 2026-02-13 13:30:00 | 497.40 | 495.63 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 498.30 | 495.15 | 0.00 | ORB-long ORB[492.65,497.50] vol=3.0x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 496.83 | 495.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 508.50 | 506.48 | 0.00 | ORB-long ORB[502.00,508.00] vol=2.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-17 10:10:00 | 506.78 | 507.15 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 514.55 | 510.42 | 0.00 | ORB-long ORB[504.05,508.15] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-20 12:10:00 | 513.10 | 510.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 514.80 | 511.72 | 0.00 | ORB-long ORB[509.00,514.30] vol=3.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:40:00 | 517.39 | 513.06 | 0.00 | T1 1.5R @ 517.39 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 514.80 | 513.31 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 533.95 | 527.79 | 0.00 | ORB-long ORB[522.00,528.45] vol=3.8x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:55:00 | 538.07 | 530.52 | 0.00 | T1 1.5R @ 538.07 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 533.95 | 531.75 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 556.00 | 551.39 | 0.00 | ORB-long ORB[548.00,554.95] vol=3.5x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 553.77 | 551.52 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 556.55 | 552.69 | 0.00 | ORB-long ORB[544.85,552.80] vol=3.2x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-03-11 09:50:00 | 554.45 | 553.22 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:05:00 | 535.20 | 542.39 | 0.00 | ORB-short ORB[541.80,548.00] vol=1.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 531.84 | 541.10 | 0.00 | T1 1.5R @ 531.84 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 535.20 | 540.09 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:50:00 | 508.60 | 511.10 | 0.00 | ORB-short ORB[509.15,516.50] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-03-27 11:30:00 | 510.82 | 510.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 09:35:00 | 510.45 | 512.94 | 0.00 | ORB-short ORB[511.40,517.60] vol=1.7x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 10:00:00 | 506.25 | 511.28 | 0.00 | T1 1.5R @ 506.25 |
| Target hit | 2026-04-02 11:15:00 | 509.25 | 508.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 593.95 | 590.32 | 0.00 | ORB-long ORB[585.95,592.00] vol=4.5x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 591.83 | 590.83 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:10:00 | 595.85 | 599.58 | 0.00 | ORB-short ORB[597.60,604.00] vol=2.2x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:05:00 | 593.17 | 599.05 | 0.00 | T1 1.5R @ 593.17 |
| Stop hit — per-position SL triggered | 2026-04-17 13:10:00 | 595.85 | 598.66 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 597.60 | 594.44 | 0.00 | ORB-long ORB[589.15,596.75] vol=2.5x ATR=2.15 |
| Stop hit — per-position SL triggered | 2026-04-23 10:30:00 | 595.45 | 594.82 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 623.95 | 619.27 | 0.00 | ORB-long ORB[614.40,621.25] vol=2.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 621.71 | 619.34 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 598.75 | 602.55 | 0.00 | ORB-short ORB[600.70,608.50] vol=1.7x ATR=2.90 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 601.65 | 601.36 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 590.30 | 591.64 | 0.00 | ORB-short ORB[590.80,597.60] vol=1.9x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:20:00 | 587.24 | 591.11 | 0.00 | T1 1.5R @ 587.24 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 590.30 | 589.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:05:00 | 521.55 | 2026-02-10 11:35:00 | 519.51 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-13 11:00:00 | 497.40 | 2026-02-13 12:25:00 | 500.46 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-13 11:00:00 | 497.40 | 2026-02-13 13:30:00 | 497.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 11:05:00 | 498.30 | 2026-02-16 11:25:00 | 496.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 09:45:00 | 508.50 | 2026-02-17 10:10:00 | 506.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 10:55:00 | 514.55 | 2026-02-20 12:10:00 | 513.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 10:25:00 | 514.80 | 2026-02-24 10:40:00 | 517.39 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-24 10:25:00 | 514.80 | 2026-02-24 11:00:00 | 514.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:45:00 | 533.95 | 2026-03-05 10:55:00 | 538.07 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-05 10:45:00 | 533.95 | 2026-03-05 11:45:00 | 533.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:45:00 | 556.00 | 2026-03-10 10:50:00 | 553.77 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-11 09:40:00 | 556.55 | 2026-03-11 09:50:00 | 554.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-13 10:05:00 | 535.20 | 2026-03-13 10:20:00 | 531.84 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-13 10:05:00 | 535.20 | 2026-03-13 10:25:00 | 535.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:50:00 | 508.60 | 2026-03-27 11:30:00 | 510.82 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-02 09:35:00 | 510.45 | 2026-04-02 10:00:00 | 506.25 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2026-04-02 09:35:00 | 510.45 | 2026-04-02 11:15:00 | 509.25 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-04-16 09:35:00 | 593.95 | 2026-04-16 09:40:00 | 591.83 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-17 11:10:00 | 595.85 | 2026-04-17 12:05:00 | 593.17 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-17 11:10:00 | 595.85 | 2026-04-17 13:10:00 | 595.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:15:00 | 597.60 | 2026-04-23 10:30:00 | 595.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-28 11:00:00 | 623.95 | 2026-04-28 11:05:00 | 621.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-29 09:40:00 | 598.75 | 2026-04-29 10:15:00 | 601.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-05 10:35:00 | 590.30 | 2026-05-05 11:20:00 | 587.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-05 10:35:00 | 590.30 | 2026-05-05 12:50:00 | 590.30 | STOP_HIT | 0.50 | 0.00% |
