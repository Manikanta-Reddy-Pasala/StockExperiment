# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3033.00
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 5
- **Avg / median % per leg:** 0.12% / 0.26%
- **Sum % (uncompounded):** 2.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.10% | 1.4% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.10% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.22% | 0.7% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.22% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 9 | 52.9% | 4 | 8 | 5 | 0.12% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:15:00 | 2906.30 | 2892.72 | 0.00 | ORB-long ORB[2864.90,2897.40] vol=1.8x ATR=9.26 |
| Stop hit — per-position SL triggered | 2026-02-11 12:55:00 | 2897.04 | 2899.15 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 2898.80 | 2882.31 | 0.00 | ORB-long ORB[2851.10,2885.00] vol=3.8x ATR=8.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:30:00 | 2912.24 | 2894.44 | 0.00 | T1 1.5R @ 2912.24 |
| Target hit | 2026-02-13 13:50:00 | 2911.00 | 2916.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 2963.40 | 2968.03 | 0.00 | ORB-short ORB[2963.60,2988.00] vol=4.9x ATR=6.30 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 2969.70 | 2967.68 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 3042.60 | 3035.09 | 0.00 | ORB-long ORB[3014.20,3041.70] vol=1.8x ATR=8.62 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 3033.98 | 3037.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 3054.00 | 3033.59 | 0.00 | ORB-long ORB[3010.40,3039.50] vol=1.7x ATR=8.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 09:55:00 | 3067.32 | 3041.30 | 0.00 | T1 1.5R @ 3067.32 |
| Target hit | 2026-03-11 13:35:00 | 3062.00 | 3062.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 2749.80 | 2732.87 | 0.00 | ORB-long ORB[2699.00,2730.00] vol=4.7x ATR=8.38 |
| Stop hit — per-position SL triggered | 2026-04-09 10:25:00 | 2741.42 | 2735.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 2786.70 | 2772.69 | 0.00 | ORB-long ORB[2750.00,2774.90] vol=1.5x ATR=10.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:00:00 | 2801.92 | 2786.76 | 0.00 | T1 1.5R @ 2801.92 |
| Target hit | 2026-04-10 13:30:00 | 2806.70 | 2806.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 2817.00 | 2801.91 | 0.00 | ORB-long ORB[2760.00,2790.00] vol=4.2x ATR=10.23 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 2806.77 | 2807.23 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 2779.70 | 2788.83 | 0.00 | ORB-short ORB[2785.90,2818.00] vol=2.3x ATR=9.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 15:00:00 | 2766.20 | 2784.16 | 0.00 | T1 1.5R @ 2766.20 |
| Target hit | 2026-04-24 15:20:00 | 2769.20 | 2781.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 2814.90 | 2790.63 | 0.00 | ORB-long ORB[2754.20,2792.30] vol=3.3x ATR=10.34 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 2804.56 | 2802.96 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 2814.00 | 2794.86 | 0.00 | ORB-long ORB[2760.00,2786.00] vol=3.9x ATR=7.47 |
| Stop hit — per-position SL triggered | 2026-04-29 10:05:00 | 2806.53 | 2798.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:35:00 | 2825.30 | 2813.24 | 0.00 | ORB-long ORB[2790.00,2814.00] vol=2.7x ATR=9.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:40:00 | 2838.89 | 2818.31 | 0.00 | T1 1.5R @ 2838.89 |
| Stop hit — per-position SL triggered | 2026-04-30 09:50:00 | 2825.30 | 2823.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:15:00 | 2906.30 | 2026-02-11 12:55:00 | 2897.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-13 09:40:00 | 2898.80 | 2026-02-13 10:30:00 | 2912.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-13 09:40:00 | 2898.80 | 2026-02-13 13:50:00 | 2911.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-25 09:35:00 | 2963.40 | 2026-02-25 09:45:00 | 2969.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-26 10:40:00 | 3042.60 | 2026-02-26 11:30:00 | 3033.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-11 09:45:00 | 3054.00 | 2026-03-11 09:55:00 | 3067.32 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-11 09:45:00 | 3054.00 | 2026-03-11 13:35:00 | 3062.00 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-09 10:15:00 | 2749.80 | 2026-04-09 10:25:00 | 2741.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-10 09:35:00 | 2786.70 | 2026-04-10 10:00:00 | 2801.92 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-10 09:35:00 | 2786.70 | 2026-04-10 13:30:00 | 2806.70 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-23 09:45:00 | 2817.00 | 2026-04-23 10:10:00 | 2806.77 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-24 11:00:00 | 2779.70 | 2026-04-24 15:00:00 | 2766.20 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-24 11:00:00 | 2779.70 | 2026-04-24 15:20:00 | 2769.20 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-27 09:45:00 | 2814.90 | 2026-04-27 10:05:00 | 2804.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-29 09:45:00 | 2814.00 | 2026-04-29 10:05:00 | 2806.53 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-30 09:35:00 | 2825.30 | 2026-04-30 09:40:00 | 2838.89 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-30 09:35:00 | 2825.30 | 2026-04-30 09:50:00 | 2825.30 | STOP_HIT | 0.50 | 0.00% |
