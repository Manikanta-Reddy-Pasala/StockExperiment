# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 875.10
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 7
- **Avg / median % per leg:** 0.39% / 0.41%
- **Sum % (uncompounded):** 7.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 8 | 57.1% | 3 | 6 | 5 | 0.36% | 5.0% |
| BUY @ 2nd Alert (retest1) | 14 | 8 | 57.1% | 3 | 6 | 5 | 0.36% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.47% | 2.4% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.47% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 11 | 57.9% | 4 | 8 | 7 | 0.39% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 687.20 | 690.01 | 0.00 | ORB-short ORB[688.15,694.95] vol=1.8x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 684.41 | 688.84 | 0.00 | T1 1.5R @ 684.41 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 687.20 | 687.75 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 691.05 | 688.44 | 0.00 | ORB-long ORB[684.85,690.00] vol=4.0x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 689.35 | 690.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 694.00 | 691.02 | 0.00 | ORB-long ORB[687.40,693.00] vol=1.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 696.23 | 691.96 | 0.00 | T1 1.5R @ 696.23 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 694.00 | 692.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 727.10 | 719.05 | 0.00 | ORB-long ORB[713.00,717.80] vol=4.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:25:00 | 730.90 | 721.62 | 0.00 | T1 1.5R @ 730.90 |
| Stop hit — per-position SL triggered | 2026-02-26 10:45:00 | 727.10 | 724.66 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 721.10 | 722.93 | 0.00 | ORB-short ORB[722.30,728.70] vol=1.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 722.98 | 722.76 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 718.20 | 711.61 | 0.00 | ORB-long ORB[703.30,714.00] vol=2.4x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-03-10 09:45:00 | 714.85 | 712.57 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 737.35 | 733.64 | 0.00 | ORB-long ORB[725.65,733.90] vol=2.4x ATR=2.90 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 734.45 | 736.92 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 710.60 | 703.06 | 0.00 | ORB-long ORB[697.00,704.55] vol=1.8x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 707.80 | 704.02 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:35:00 | 667.00 | 670.12 | 0.00 | ORB-short ORB[667.30,675.50] vol=2.0x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:50:00 | 662.25 | 668.74 | 0.00 | T1 1.5R @ 662.25 |
| Target hit | 2026-03-23 13:45:00 | 657.00 | 655.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2026-03-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:55:00 | 682.80 | 679.60 | 0.00 | ORB-long ORB[675.00,681.55] vol=1.8x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:35:00 | 687.57 | 681.62 | 0.00 | T1 1.5R @ 687.57 |
| Target hit | 2026-03-25 15:20:00 | 692.80 | 685.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 738.50 | 734.64 | 0.00 | ORB-long ORB[726.50,737.30] vol=3.1x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:05:00 | 743.30 | 738.48 | 0.00 | T1 1.5R @ 743.30 |
| Target hit | 2026-04-15 15:20:00 | 748.30 | 742.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:20:00 | 818.00 | 808.43 | 0.00 | ORB-long ORB[804.05,810.95] vol=3.0x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:25:00 | 823.04 | 810.73 | 0.00 | T1 1.5R @ 823.04 |
| Target hit | 2026-04-22 15:20:00 | 825.55 | 821.97 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:30:00 | 687.20 | 2026-02-10 09:40:00 | 684.41 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-10 09:30:00 | 687.20 | 2026-02-10 10:25:00 | 687.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:40:00 | 691.05 | 2026-02-17 10:05:00 | 689.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-18 10:55:00 | 694.00 | 2026-02-18 11:10:00 | 696.23 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-18 10:55:00 | 694.00 | 2026-02-18 11:15:00 | 694.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:20:00 | 727.10 | 2026-02-26 10:25:00 | 730.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-26 10:20:00 | 727.10 | 2026-02-26 10:45:00 | 727.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:15:00 | 721.10 | 2026-02-27 10:25:00 | 722.98 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-10 09:40:00 | 718.20 | 2026-03-10 09:45:00 | 714.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-11 09:30:00 | 737.35 | 2026-03-11 11:10:00 | 734.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-17 11:05:00 | 710.60 | 2026-03-17 11:20:00 | 707.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-23 09:35:00 | 667.00 | 2026-03-23 09:50:00 | 662.25 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-23 09:35:00 | 667.00 | 2026-03-23 13:45:00 | 657.00 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2026-03-25 09:55:00 | 682.80 | 2026-03-25 12:35:00 | 687.57 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-25 09:55:00 | 682.80 | 2026-03-25 15:20:00 | 692.80 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2026-04-15 09:40:00 | 738.50 | 2026-04-15 10:05:00 | 743.30 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-15 09:40:00 | 738.50 | 2026-04-15 15:20:00 | 748.30 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2026-04-22 10:20:00 | 818.00 | 2026-04-22 10:25:00 | 823.04 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-22 10:20:00 | 818.00 | 2026-04-22 15:20:00 | 825.55 | TARGET_HIT | 0.50 | 0.92% |
