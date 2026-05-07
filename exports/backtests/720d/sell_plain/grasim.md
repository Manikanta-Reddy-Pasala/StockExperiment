# GRASIM (GRASIM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 6 |
| PENDING | 17 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -1.80% / -1.90%
- **Sum % (uncompounded):** -21.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 2 | 16.7% | 0 | 12 | 0 | -1.80% | -21.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 0 | 12 | 0 | -1.80% | -21.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 2 | 16.7% | 0 | 12 | 0 | -1.80% | -21.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 2628.00 | 2690.12 | 2690.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 2583.90 | 2685.30 | 2687.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 2622.10 | 2611.64 | 2643.46 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 2607.45 | 2612.40 | 2641.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2607.45 | 2612.40 | 2641.85 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-27 11:15:00 | 2598.00 | 2612.15 | 2641.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-27 12:15:00 | 2627.85 | 2612.30 | 2641.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-28 11:15:00 | 2589.60 | 2612.31 | 2640.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:15:00 | 2562.60 | 2611.56 | 2639.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 2657.35 | 2611.06 | 2638.21 | SL hit (close>static) qty=1.00 sl=2649.45 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-17 14:15:00 | 2597.80 | 2648.44 | 2652.29 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-18 09:15:00 | 2602.90 | 2647.50 | 2651.78 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2024-12-18 10:15:00 | 2590.00 | 2646.93 | 2651.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:15:00 | 2599.65 | 2645.94 | 2650.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-27 15:15:00 | 2596.10 | 2460.70 | 2465.89 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-28 09:15:00 | 2609.20 | 2462.18 | 2466.60 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 2653.75 | 2494.48 | 2483.18 | SL hit (close>static) qty=1.00 sl=2649.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 2510.45 | 2509.28 | 2491.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.59 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 2652.90 | 2520.68 | 2499.33 | SL hit (close>static) qty=1.00 sl=2649.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-28 10:15:00 | 2594.00 | 2685.35 | 2638.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:15:00 | 2588.90 | 2683.39 | 2638.04 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2724.60 | 2631.42 | 2619.61 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-10 09:15:00 | 2724.60 | 2631.42 | 2619.61 | SL hit (close>static) qty=1.00 sl=2649.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-13 14:15:00 | 2660.30 | 2646.84 | 2629.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 2652.90 | 2647.08 | 2629.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 2752.10 | 2665.35 | 2642.88 | SL hit (close>static) qty=1.00 sl=2742.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.65 | 2803.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 2723.00 | 2800.52 | 2802.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.38 | 2778.39 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.34 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-09 13:15:00 | 2761.30 | 2762.44 | 2778.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2745.60 | 2762.14 | 2777.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-12-10 12:15:00 | 2748.20 | 2762.17 | 2777.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 2746.70 | 2761.86 | 2777.34 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.20 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.20 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-12 09:15:00 | 2749.00 | 2812.60 | 2803.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 2761.20 | 2811.49 | 2802.68 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 2780.50 | 2811.18 | 2802.57 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-20 11:15:00 | 2763.20 | 2806.63 | 2801.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:15:00 | 2747.60 | 2805.59 | 2801.00 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 2788.00 | 2798.89 | 2797.77 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2775.00 | 2798.65 | 2797.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-22 12:15:00 | 2770.40 | 2798.11 | 2797.40 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-22 14:15:00 | 2792.20 | 2797.74 | 2797.22 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-01-23 11:15:00 | 2771.00 | 2797.17 | 2796.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-23 12:15:00 | 2772.80 | 2796.92 | 2796.82 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.54 | 2796.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 2741.70 | 2803.90 | 2800.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.88 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.88 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 12:15:00 | 2811.50 | 2800.82 | 2799.29 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:15:00 | 2806.90 | 2800.99 | 2799.39 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 2871.10 | 2807.33 | 2802.79 | SL hit (close>static) qty=1.00 sl=2869.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-27 10:15:00 | 2810.90 | 2860.22 | 2839.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 2796.90 | 2858.93 | 2838.65 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 2796.00 | 2857.91 | 2838.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2752.90 | 2856.23 | 2837.69 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 4020m) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2630.00 | 2793.23 | 2806.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-28 13:15:00 | 2562.60 | 2024-12-02 09:15:00 | 2657.35 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-12-18 12:15:00 | 2599.65 | 2025-04-03 11:15:00 | 2653.75 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-07 11:15:00 | 2485.00 | 2025-04-11 11:15:00 | 2652.90 | STOP_HIT | 1.00 | -6.76% |
| SELL | retest2 | 2025-05-28 12:15:00 | 2588.90 | 2025-06-10 09:15:00 | 2724.60 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-06-16 09:15:00 | 2652.90 | 2025-06-24 09:15:00 | 2752.10 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2745.60 | 2025-12-11 13:15:00 | 2797.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-10 14:15:00 | 2746.70 | 2025-12-11 13:15:00 | 2797.90 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-01-12 11:15:00 | 2761.20 | 2026-01-12 12:15:00 | 2780.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-20 13:15:00 | 2747.60 | 2026-01-22 09:15:00 | 2788.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-03 14:15:00 | 2806.90 | 2026-02-05 15:15:00 | 2871.10 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-27 12:15:00 | 2796.90 | 2026-03-09 09:15:00 | 2661.50 | STOP_HIT | 1.00 | 4.84% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2752.90 | 2026-03-09 09:15:00 | 2661.50 | STOP_HIT | 1.00 | 3.32% |
