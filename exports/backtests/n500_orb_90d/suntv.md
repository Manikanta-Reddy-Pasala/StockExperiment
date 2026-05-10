# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 572.15
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
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 5
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 3.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.34% | 3.4% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.34% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.18% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 588.10 | 584.01 | 0.00 | ORB-long ORB[573.50,579.80] vol=5.2x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:35:00 | 591.71 | 586.65 | 0.00 | T1 1.5R @ 591.71 |
| Target hit | 2026-02-17 11:40:00 | 589.35 | 589.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 588.70 | 593.88 | 0.00 | ORB-short ORB[592.00,599.35] vol=1.5x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:30:00 | 585.54 | 591.05 | 0.00 | T1 1.5R @ 585.54 |
| Stop hit — per-position SL triggered | 2026-02-18 12:40:00 | 588.70 | 590.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:10:00 | 585.60 | 587.93 | 0.00 | ORB-short ORB[586.70,593.20] vol=2.4x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 587.20 | 587.73 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 581.15 | 584.76 | 0.00 | ORB-short ORB[584.65,590.45] vol=3.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-02-23 12:45:00 | 582.72 | 583.90 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 599.25 | 604.19 | 0.00 | ORB-short ORB[602.40,609.60] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 601.29 | 604.12 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 617.35 | 614.07 | 0.00 | ORB-long ORB[609.75,615.55] vol=1.6x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:35:00 | 622.34 | 626.17 | 0.00 | T1 1.5R @ 622.34 |
| Target hit | 2026-02-27 09:55:00 | 635.25 | 635.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 566.10 | 562.38 | 0.00 | ORB-long ORB[553.20,559.90] vol=6.8x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 562.73 | 562.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:40:00 | 612.70 | 604.95 | 0.00 | ORB-long ORB[599.35,607.95] vol=1.8x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-04-08 11:05:00 | 609.79 | 605.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 626.95 | 623.30 | 0.00 | ORB-long ORB[619.30,625.00] vol=2.3x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:55:00 | 630.21 | 625.15 | 0.00 | T1 1.5R @ 630.21 |
| Stop hit — per-position SL triggered | 2026-04-15 11:35:00 | 626.95 | 628.15 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 644.65 | 651.60 | 0.00 | ORB-short ORB[648.80,655.65] vol=2.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 647.03 | 651.17 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 600.75 | 606.36 | 0.00 | ORB-short ORB[605.25,612.55] vol=2.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:40:00 | 597.48 | 603.31 | 0.00 | T1 1.5R @ 597.48 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 600.75 | 598.99 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 602.00 | 597.61 | 0.00 | ORB-long ORB[592.00,600.70] vol=5.3x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 600.22 | 597.68 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 577.50 | 572.71 | 0.00 | ORB-long ORB[569.70,576.00] vol=4.9x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-05-07 11:10:00 | 576.04 | 572.98 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:10:00 | 588.10 | 2026-02-17 10:35:00 | 591.71 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-17 10:10:00 | 588.10 | 2026-02-17 11:40:00 | 589.35 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-02-18 09:55:00 | 588.70 | 2026-02-18 12:30:00 | 585.54 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-18 09:55:00 | 588.70 | 2026-02-18 12:40:00 | 588.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:10:00 | 585.60 | 2026-02-19 10:15:00 | 587.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-23 11:15:00 | 581.15 | 2026-02-23 12:45:00 | 582.72 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-26 10:55:00 | 599.25 | 2026-02-26 11:05:00 | 601.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-27 09:30:00 | 617.35 | 2026-02-27 09:35:00 | 622.34 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-02-27 09:30:00 | 617.35 | 2026-02-27 09:55:00 | 635.25 | TARGET_HIT | 0.50 | 2.90% |
| BUY | retest1 | 2026-03-17 10:00:00 | 566.10 | 2026-03-17 10:35:00 | 562.73 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-08 10:40:00 | 612.70 | 2026-04-08 11:05:00 | 609.79 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-15 09:45:00 | 626.95 | 2026-04-15 09:55:00 | 630.21 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-15 09:45:00 | 626.95 | 2026-04-15 11:35:00 | 626.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 10:35:00 | 644.65 | 2026-04-21 10:40:00 | 647.03 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-28 09:30:00 | 600.75 | 2026-04-28 09:40:00 | 597.48 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-28 09:30:00 | 600.75 | 2026-04-28 11:45:00 | 600.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 11:15:00 | 602.00 | 2026-04-29 11:20:00 | 600.22 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-07 11:05:00 | 577.50 | 2026-05-07 11:10:00 | 576.04 | STOP_HIT | 1.00 | -0.25% |
