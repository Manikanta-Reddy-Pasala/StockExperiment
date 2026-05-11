# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2024-09-09 09:15:00 → 2026-05-08 15:25:00 (29350 bars)
- **Last close:** 4424.00
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
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 7
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 4.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.25% | 4.7% |
| BUY @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.25% | 4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.09% | -0.6% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.09% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 4 | 14 | 7 | 0.16% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 2769.10 | 2735.78 | 0.00 | ORB-long ORB[2711.55,2751.95] vol=2.7x ATR=16.32 |
| Stop hit — per-position SL triggered | 2024-09-12 09:40:00 | 2752.78 | 2741.24 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-09-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:50:00 | 2805.65 | 2761.61 | 0.00 | ORB-long ORB[2702.00,2740.00] vol=9.9x ATR=17.69 |
| Stop hit — per-position SL triggered | 2024-09-16 09:55:00 | 2787.96 | 2769.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:50:00 | 2797.35 | 2774.89 | 0.00 | ORB-long ORB[2751.05,2785.00] vol=2.9x ATR=13.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:55:00 | 2817.87 | 2784.07 | 0.00 | T1 1.5R @ 2817.87 |
| Stop hit — per-position SL triggered | 2024-09-18 10:00:00 | 2797.35 | 2786.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 2653.80 | 2704.50 | 0.00 | ORB-short ORB[2704.95,2741.80] vol=2.0x ATR=17.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:00:00 | 2628.30 | 2695.00 | 0.00 | T1 1.5R @ 2628.30 |
| Target hit | 2024-09-19 14:50:00 | 2646.20 | 2641.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 2498.50 | 2517.68 | 0.00 | ORB-short ORB[2513.35,2548.00] vol=2.6x ATR=9.27 |
| Stop hit — per-position SL triggered | 2024-10-01 09:35:00 | 2507.77 | 2514.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 2482.00 | 2449.59 | 0.00 | ORB-long ORB[2421.05,2456.75] vol=2.3x ATR=12.43 |
| Stop hit — per-position SL triggered | 2024-10-11 10:00:00 | 2469.57 | 2454.68 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 2867.00 | 2842.24 | 0.00 | ORB-long ORB[2811.05,2845.90] vol=2.5x ATR=14.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:40:00 | 2888.20 | 2853.50 | 0.00 | T1 1.5R @ 2888.20 |
| Stop hit — per-position SL triggered | 2024-11-12 09:45:00 | 2867.00 | 2860.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 2889.95 | 2848.49 | 0.00 | ORB-long ORB[2815.55,2857.00] vol=3.7x ATR=13.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:00:00 | 2910.25 | 2898.95 | 0.00 | T1 1.5R @ 2910.25 |
| Target hit | 2024-11-27 10:40:00 | 2932.00 | 2974.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 2771.10 | 2749.79 | 0.00 | ORB-long ORB[2730.00,2762.95] vol=1.7x ATR=13.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:35:00 | 2791.51 | 2763.42 | 0.00 | T1 1.5R @ 2791.51 |
| Target hit | 2024-12-02 10:35:00 | 2776.95 | 2782.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 2835.00 | 2811.81 | 0.00 | ORB-long ORB[2787.40,2815.95] vol=3.7x ATR=13.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:45:00 | 2854.53 | 2819.42 | 0.00 | T1 1.5R @ 2854.53 |
| Target hit | 2024-12-03 12:05:00 | 2895.00 | 2902.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-12-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:50:00 | 2886.05 | 2870.08 | 0.00 | ORB-long ORB[2843.95,2886.00] vol=1.7x ATR=13.82 |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 2872.23 | 2873.61 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 2766.10 | 2800.67 | 0.00 | ORB-short ORB[2792.25,2829.00] vol=3.7x ATR=12.99 |
| Stop hit — per-position SL triggered | 2024-12-13 09:35:00 | 2779.09 | 2794.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:40:00 | 2834.45 | 2816.92 | 0.00 | ORB-long ORB[2775.00,2814.20] vol=1.7x ATR=16.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:00:00 | 2858.71 | 2827.14 | 0.00 | T1 1.5R @ 2858.71 |
| Stop hit — per-position SL triggered | 2024-12-16 10:10:00 | 2834.45 | 2829.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 2814.00 | 2838.19 | 0.00 | ORB-short ORB[2840.90,2880.00] vol=1.6x ATR=13.80 |
| Stop hit — per-position SL triggered | 2024-12-17 10:25:00 | 2827.80 | 2837.05 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 2663.80 | 2689.12 | 0.00 | ORB-short ORB[2686.85,2723.65] vol=4.6x ATR=12.49 |
| Stop hit — per-position SL triggered | 2024-12-26 09:45:00 | 2676.29 | 2685.56 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 2918.40 | 2896.12 | 0.00 | ORB-long ORB[2866.60,2899.00] vol=3.2x ATR=15.98 |
| Stop hit — per-position SL triggered | 2025-01-02 09:40:00 | 2902.42 | 2902.09 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 2795.00 | 2785.21 | 0.00 | ORB-long ORB[2763.00,2791.80] vol=1.8x ATR=9.42 |
| Stop hit — per-position SL triggered | 2025-01-09 11:05:00 | 2785.58 | 2785.87 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:50:00 | 1546.10 | 1529.96 | 0.00 | ORB-long ORB[1512.00,1534.90] vol=3.5x ATR=7.66 |
| Stop hit — per-position SL triggered | 2025-04-23 10:00:00 | 1538.44 | 1533.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-12 09:35:00 | 2769.10 | 2024-09-12 09:40:00 | 2752.78 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-09-16 09:50:00 | 2805.65 | 2024-09-16 09:55:00 | 2787.96 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-09-18 09:50:00 | 2797.35 | 2024-09-18 09:55:00 | 2817.87 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-09-18 09:50:00 | 2797.35 | 2024-09-18 10:00:00 | 2797.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:45:00 | 2653.80 | 2024-09-19 10:00:00 | 2628.30 | PARTIAL | 0.50 | 0.96% |
| SELL | retest1 | 2024-09-19 09:45:00 | 2653.80 | 2024-09-19 14:50:00 | 2646.20 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-01 09:30:00 | 2498.50 | 2024-10-01 09:35:00 | 2507.77 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-11 09:55:00 | 2482.00 | 2024-10-11 10:00:00 | 2469.57 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-12 09:30:00 | 2867.00 | 2024-11-12 09:40:00 | 2888.20 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-11-12 09:30:00 | 2867.00 | 2024-11-12 09:45:00 | 2867.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:55:00 | 2889.95 | 2024-11-27 10:00:00 | 2910.25 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-11-27 09:55:00 | 2889.95 | 2024-11-27 10:40:00 | 2932.00 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2024-12-02 09:30:00 | 2771.10 | 2024-12-02 09:35:00 | 2791.51 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-12-02 09:30:00 | 2771.10 | 2024-12-02 10:35:00 | 2776.95 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-12-03 10:40:00 | 2835.00 | 2024-12-03 10:45:00 | 2854.53 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-12-03 10:40:00 | 2835.00 | 2024-12-03 12:05:00 | 2895.00 | TARGET_HIT | 0.50 | 2.12% |
| BUY | retest1 | 2024-12-05 09:50:00 | 2886.05 | 2024-12-05 10:15:00 | 2872.23 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-13 09:30:00 | 2766.10 | 2024-12-13 09:35:00 | 2779.09 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-12-16 09:40:00 | 2834.45 | 2024-12-16 10:00:00 | 2858.71 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2024-12-16 09:40:00 | 2834.45 | 2024-12-16 10:10:00 | 2834.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:20:00 | 2814.00 | 2024-12-17 10:25:00 | 2827.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-12-26 09:35:00 | 2663.80 | 2024-12-26 09:45:00 | 2676.29 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-02 09:30:00 | 2918.40 | 2025-01-02 09:40:00 | 2902.42 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-01-09 10:45:00 | 2795.00 | 2025-01-09 11:05:00 | 2785.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-23 09:50:00 | 1546.10 | 2025-04-23 10:00:00 | 1538.44 | STOP_HIT | 1.00 | -0.50% |
