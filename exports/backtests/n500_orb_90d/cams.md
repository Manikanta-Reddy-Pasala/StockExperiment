# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 835.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 2
- **Avg / median % per leg:** -0.13% / -0.30%
- **Sum % (uncompounded):** -1.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.5% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.39% | -2.3% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.39% | -2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 3 | 21.4% | 1 | 11 | 2 | -0.13% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 734.55 | 729.96 | 0.00 | ORB-long ORB[725.05,732.60] vol=3.4x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:35:00 | 739.41 | 732.98 | 0.00 | T1 1.5R @ 739.41 |
| Stop hit — per-position SL triggered | 2026-02-09 12:30:00 | 734.55 | 735.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 746.90 | 743.71 | 0.00 | ORB-long ORB[741.25,746.30] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-11 10:55:00 | 745.27 | 744.02 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 733.75 | 736.63 | 0.00 | ORB-short ORB[734.50,738.50] vol=1.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 735.99 | 735.08 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 730.85 | 734.09 | 0.00 | ORB-short ORB[731.75,736.85] vol=1.8x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 732.96 | 733.93 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 718.15 | 725.56 | 0.00 | ORB-short ORB[722.05,729.40] vol=2.4x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-02-23 10:50:00 | 720.27 | 725.26 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 657.85 | 664.02 | 0.00 | ORB-short ORB[661.10,670.80] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 660.90 | 661.77 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:40:00 | 620.90 | 626.06 | 0.00 | ORB-short ORB[621.30,629.95] vol=1.8x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-30 11:05:00 | 624.15 | 625.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 724.75 | 719.49 | 0.00 | ORB-long ORB[712.30,721.65] vol=2.5x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:20:00 | 728.98 | 722.71 | 0.00 | T1 1.5R @ 728.98 |
| Target hit | 2026-04-15 15:20:00 | 728.70 | 726.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 745.40 | 739.88 | 0.00 | ORB-long ORB[736.25,743.75] vol=2.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-04-17 10:50:00 | 742.66 | 740.29 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 756.35 | 752.72 | 0.00 | ORB-long ORB[745.85,753.20] vol=3.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-04-21 10:30:00 | 753.91 | 753.82 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 740.10 | 743.92 | 0.00 | ORB-short ORB[741.90,750.60] vol=1.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-05-04 09:40:00 | 743.49 | 743.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 840.85 | 835.43 | 0.00 | ORB-long ORB[831.30,839.00] vol=1.5x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 837.49 | 835.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 734.55 | 2026-02-09 11:35:00 | 739.41 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-02-09 10:55:00 | 734.55 | 2026-02-09 12:30:00 | 734.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:40:00 | 746.90 | 2026-02-11 10:55:00 | 745.27 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-18 09:45:00 | 733.75 | 2026-02-18 10:15:00 | 735.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-19 11:15:00 | 730.85 | 2026-02-19 11:25:00 | 732.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 10:45:00 | 718.15 | 2026-02-23 10:50:00 | 720.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-12 10:10:00 | 657.85 | 2026-03-12 11:00:00 | 660.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-30 10:40:00 | 620.90 | 2026-03-30 11:05:00 | 624.15 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-15 10:05:00 | 724.75 | 2026-04-15 10:20:00 | 728.98 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-15 10:05:00 | 724.75 | 2026-04-15 15:20:00 | 728.70 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-17 10:40:00 | 745.40 | 2026-04-17 10:50:00 | 742.66 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-21 09:40:00 | 756.35 | 2026-04-21 10:30:00 | 753.91 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-04 09:30:00 | 740.10 | 2026-05-04 09:40:00 | 743.49 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-05-08 09:40:00 | 840.85 | 2026-05-08 09:45:00 | 837.49 | STOP_HIT | 1.00 | -0.40% |
