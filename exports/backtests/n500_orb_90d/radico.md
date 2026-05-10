# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3481.90
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 3
- **Avg / median % per leg:** 0.11% / -0.26%
- **Sum % (uncompounded):** 1.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.30% | 2.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.30% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.12% | -0.7% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.12% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.11% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2787.80 | 2797.90 | 0.00 | ORB-short ORB[2788.60,2822.60] vol=1.5x ATR=7.72 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 2795.52 | 2796.43 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 2795.70 | 2778.73 | 0.00 | ORB-long ORB[2743.60,2779.00] vol=1.6x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 2807.64 | 2783.32 | 0.00 | T1 1.5R @ 2807.64 |
| Target hit | 2026-02-11 15:20:00 | 2811.00 | 2805.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:50:00 | 2856.20 | 2830.32 | 0.00 | ORB-long ORB[2785.10,2815.40] vol=4.1x ATR=11.38 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 2844.82 | 2831.33 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 2695.20 | 2707.03 | 0.00 | ORB-short ORB[2701.30,2731.70] vol=1.6x ATR=7.56 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 2702.76 | 2705.68 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 2730.30 | 2714.77 | 0.00 | ORB-long ORB[2708.00,2729.30] vol=2.1x ATR=7.17 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 2723.13 | 2715.90 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:45:00 | 2531.10 | 2547.75 | 0.00 | ORB-short ORB[2552.00,2579.90] vol=1.5x ATR=8.88 |
| Stop hit — per-position SL triggered | 2026-03-04 11:30:00 | 2539.98 | 2541.69 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 10:05:00 | 2817.80 | 2795.84 | 0.00 | ORB-long ORB[2757.10,2788.00] vol=2.7x ATR=11.51 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 2806.29 | 2797.07 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:55:00 | 2637.00 | 2663.03 | 0.00 | ORB-short ORB[2668.90,2696.20] vol=2.1x ATR=8.22 |
| Stop hit — per-position SL triggered | 2026-03-19 11:00:00 | 2645.22 | 2660.85 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 2751.10 | 2736.68 | 0.00 | ORB-long ORB[2722.20,2749.00] vol=2.1x ATR=9.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:55:00 | 2765.13 | 2741.14 | 0.00 | T1 1.5R @ 2765.13 |
| Target hit | 2026-04-10 15:20:00 | 2797.90 | 2782.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:20:00 | 3357.90 | 3379.96 | 0.00 | ORB-short ORB[3375.10,3414.20] vol=1.8x ATR=11.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:25:00 | 3340.43 | 3368.89 | 0.00 | T1 1.5R @ 3340.43 |
| Stop hit — per-position SL triggered | 2026-05-05 10:35:00 | 3357.90 | 3367.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:00:00 | 2787.80 | 2026-02-10 11:25:00 | 2795.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-11 10:50:00 | 2795.70 | 2026-02-11 11:10:00 | 2807.64 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-11 10:50:00 | 2795.70 | 2026-02-11 15:20:00 | 2811.00 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-16 09:50:00 | 2856.20 | 2026-02-16 09:55:00 | 2844.82 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-20 10:45:00 | 2695.20 | 2026-02-20 11:00:00 | 2702.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 10:45:00 | 2730.30 | 2026-02-24 11:00:00 | 2723.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-04 10:45:00 | 2531.10 | 2026-03-04 11:30:00 | 2539.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-16 10:05:00 | 2817.80 | 2026-03-16 10:10:00 | 2806.29 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-19 10:55:00 | 2637.00 | 2026-03-19 11:00:00 | 2645.22 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-10 09:45:00 | 2751.10 | 2026-04-10 09:55:00 | 2765.13 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-10 09:45:00 | 2751.10 | 2026-04-10 15:20:00 | 2797.90 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2026-05-05 10:20:00 | 3357.90 | 2026-05-05 10:25:00 | 3340.43 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-05 10:20:00 | 3357.90 | 2026-05-05 10:35:00 | 3357.90 | STOP_HIT | 0.50 | 0.00% |
