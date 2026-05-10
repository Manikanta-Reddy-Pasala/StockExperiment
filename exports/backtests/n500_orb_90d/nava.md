# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 727.65
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 7
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 1 | 10 | 5 | 0.11% | 1.8% |
| BUY @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 1 | 10 | 5 | 0.11% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.22% | 1.5% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.22% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 2 | 14 | 7 | 0.14% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 563.25 | 561.95 | 0.00 | ORB-long ORB[558.00,562.00] vol=3.0x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 561.01 | 561.96 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 559.20 | 563.45 | 0.00 | ORB-short ORB[562.65,567.70] vol=2.7x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 556.23 | 560.91 | 0.00 | T1 1.5R @ 556.23 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 559.20 | 558.44 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:15:00 | 558.35 | 554.62 | 0.00 | ORB-long ORB[549.30,555.15] vol=2.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:25:00 | 561.74 | 560.06 | 0.00 | T1 1.5R @ 561.74 |
| Target hit | 2026-02-20 11:00:00 | 564.55 | 565.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 574.25 | 575.71 | 0.00 | ORB-short ORB[575.95,578.70] vol=1.9x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 576.28 | 575.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:00:00 | 598.95 | 590.23 | 0.00 | ORB-long ORB[581.35,587.90] vol=4.2x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 596.32 | 593.07 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 564.00 | 566.31 | 0.00 | ORB-short ORB[565.00,569.95] vol=4.0x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-03-05 11:30:00 | 565.70 | 565.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 568.85 | 572.70 | 0.00 | ORB-short ORB[568.90,575.90] vol=1.9x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-03-06 11:45:00 | 571.10 | 572.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 562.75 | 567.05 | 0.00 | ORB-short ORB[565.65,572.90] vol=2.4x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 558.88 | 562.91 | 0.00 | T1 1.5R @ 558.88 |
| Target hit | 2026-03-13 15:20:00 | 555.00 | 557.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:10:00 | 549.85 | 545.34 | 0.00 | ORB-long ORB[536.35,544.00] vol=2.2x ATR=2.19 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 547.66 | 545.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 603.00 | 598.97 | 0.00 | ORB-long ORB[594.30,601.00] vol=6.1x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:10:00 | 606.73 | 601.59 | 0.00 | T1 1.5R @ 606.73 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 603.00 | 602.81 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 625.00 | 621.56 | 0.00 | ORB-long ORB[615.65,624.00] vol=1.6x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:40:00 | 629.80 | 623.64 | 0.00 | T1 1.5R @ 629.80 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 625.00 | 624.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 700.00 | 696.45 | 0.00 | ORB-long ORB[690.55,699.00] vol=2.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 697.44 | 697.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:30:00 | 712.80 | 703.84 | 0.00 | ORB-long ORB[695.05,704.85] vol=6.7x ATR=3.53 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 709.27 | 708.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 670.75 | 664.47 | 0.00 | ORB-long ORB[658.50,665.45] vol=2.2x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-04-27 09:45:00 | 667.39 | 664.98 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 675.20 | 668.18 | 0.00 | ORB-long ORB[661.25,669.90] vol=1.9x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:25:00 | 679.95 | 671.50 | 0.00 | T1 1.5R @ 679.95 |
| Stop hit — per-position SL triggered | 2026-05-04 12:40:00 | 675.20 | 673.38 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 689.75 | 685.69 | 0.00 | ORB-long ORB[677.25,683.00] vol=7.6x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:00:00 | 693.60 | 688.01 | 0.00 | T1 1.5R @ 693.60 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 689.75 | 688.09 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 09:35:00 | 563.25 | 2026-02-18 09:40:00 | 561.01 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-19 09:30:00 | 559.20 | 2026-02-19 09:50:00 | 556.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-19 09:30:00 | 559.20 | 2026-02-19 10:40:00 | 559.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:15:00 | 558.35 | 2026-02-20 10:25:00 | 561.74 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-20 10:15:00 | 558.35 | 2026-02-20 11:00:00 | 564.55 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2026-02-25 10:15:00 | 574.25 | 2026-02-25 10:40:00 | 576.28 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 10:00:00 | 598.95 | 2026-02-26 10:20:00 | 596.32 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-05 10:55:00 | 564.00 | 2026-03-05 11:30:00 | 565.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-06 10:45:00 | 568.85 | 2026-03-06 11:45:00 | 571.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-13 09:50:00 | 562.75 | 2026-03-13 10:20:00 | 558.88 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-13 09:50:00 | 562.75 | 2026-03-13 15:20:00 | 555.00 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2026-03-17 11:10:00 | 549.85 | 2026-03-17 11:25:00 | 547.66 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-10 10:05:00 | 603.00 | 2026-04-10 10:10:00 | 606.73 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-10 10:05:00 | 603.00 | 2026-04-10 11:45:00 | 603.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:35:00 | 625.00 | 2026-04-15 09:40:00 | 629.80 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-04-15 09:35:00 | 625.00 | 2026-04-15 09:45:00 | 625.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 700.00 | 2026-04-22 09:55:00 | 697.44 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-23 10:30:00 | 712.80 | 2026-04-23 10:35:00 | 709.27 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-27 09:40:00 | 670.75 | 2026-04-27 09:45:00 | 667.39 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-04 11:00:00 | 675.20 | 2026-05-04 11:25:00 | 679.95 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-05-04 11:00:00 | 675.20 | 2026-05-04 12:40:00 | 675.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:30:00 | 689.75 | 2026-05-06 10:00:00 | 693.60 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-06 09:30:00 | 689.75 | 2026-05-06 10:05:00 | 689.75 | STOP_HIT | 0.50 | 0.00% |
