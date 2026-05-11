# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2214.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 2 |
| TARGET_HIT | 7 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 18
- **Target hits / Stop hits / Partials:** 7 / 18 / 2
- **Avg / median % per leg:** 1.67% / -0.83%
- **Sum % (uncompounded):** 45.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 5 | 21.7% | 5 | 18 | 0 | 0.66% | 15.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 5 | 21.7% | 5 | 18 | 0 | 0.66% | 15.2% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 9 | 33.3% | 7 | 18 | 2 | 1.67% | 45.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 2557.50 | 2493.60 | 2493.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 2565.10 | 2494.31 | 2493.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 2496.10 | 2498.71 | 2496.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:30:00 | 2488.80 | 2498.20 | 2495.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 2487.10 | 2498.20 | 2495.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:45:00 | 2487.80 | 2498.15 | 2495.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2502.50 | 2498.09 | 2495.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 2534.00 | 2498.28 | 2496.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 2486.60 | 2500.25 | 2497.19 | SL hit (close<static) qty=1.00 sl=2490.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2740.00 | 2780.22 | 2780.30 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2780.48 | 2780.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2782.39 | 2781.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 2793.60 | 2804.16 | 2793.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 2770.80 | 2803.83 | 2793.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 2770.80 | 2803.83 | 2793.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2762.50 | 2803.42 | 2793.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 2762.50 | 2803.42 | 2793.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2763.10 | 2798.99 | 2791.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 2763.10 | 2798.99 | 2791.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2775.00 | 2790.98 | 2788.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 2775.00 | 2790.98 | 2788.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 2777.40 | 2790.85 | 2787.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 2778.90 | 2790.85 | 2787.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2775.70 | 2790.68 | 2787.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 2775.70 | 2790.68 | 2787.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2775.00 | 2790.52 | 2787.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2767.30 | 2790.30 | 2787.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2775.20 | 2790.15 | 2787.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 2787.90 | 2789.59 | 2787.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:30:00 | 2785.10 | 2794.51 | 2790.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 2749.10 | 2794.06 | 2790.01 | SL hit (close<static) qty=1.00 sl=2762.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2662.50 | 2785.91 | 2786.07 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 2819.30 | 2783.43 | 2783.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 2846.00 | 2784.90 | 2784.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 2852.50 | 2856.65 | 2830.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:00:00 | 2852.50 | 2856.65 | 2830.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 2824.50 | 2856.13 | 2831.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 2824.50 | 2856.13 | 2831.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 2818.00 | 2855.75 | 2830.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:15:00 | 2801.50 | 2855.75 | 2830.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 2820.00 | 2844.81 | 2827.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:30:00 | 2826.60 | 2844.42 | 2827.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2805.30 | 2843.79 | 2827.49 | SL hit (close<static) qty=1.00 sl=2812.20 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 2753.40 | 2822.74 | 2822.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 2735.00 | 2819.92 | 2821.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 2830.70 | 2812.18 | 2817.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:45:00 | 2830.50 | 2812.35 | 2817.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:15:00 | 2689.16 | 2811.27 | 2816.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:15:00 | 2688.97 | 2811.27 | 2816.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-11 12:15:00 | 2547.63 | 2750.78 | 2782.85 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-02 10:45:00 | 2496.10 | 2025-06-05 10:15:00 | 2486.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-06-02 13:30:00 | 2488.80 | 2025-06-25 10:15:00 | 2737.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 14:15:00 | 2487.10 | 2025-06-25 10:15:00 | 2735.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 14:45:00 | 2487.80 | 2025-06-25 10:15:00 | 2736.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 2534.00 | 2025-06-25 12:15:00 | 2745.71 | TARGET_HIT | 1.00 | 8.35% |
| BUY | retest2 | 2025-06-05 11:45:00 | 2518.90 | 2025-06-26 09:15:00 | 2770.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 15:15:00 | 2787.90 | 2025-11-14 12:15:00 | 2749.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-11-14 11:30:00 | 2785.10 | 2025-11-14 12:15:00 | 2749.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-01 14:30:00 | 2826.60 | 2026-01-02 09:15:00 | 2805.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-06 11:45:00 | 2824.60 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-01-07 09:15:00 | 2829.50 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-07 09:45:00 | 2825.60 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-09 09:30:00 | 2827.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-01-09 10:15:00 | 2837.80 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-01-12 09:45:00 | 2827.30 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-12 10:15:00 | 2826.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-13 10:45:00 | 2862.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-13 14:45:00 | 2865.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-01-16 09:15:00 | 2899.90 | 2026-01-21 09:15:00 | 2764.10 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-01-19 10:00:00 | 2867.70 | 2026-01-21 09:15:00 | 2764.10 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-22 09:15:00 | 2854.50 | 2026-01-22 14:15:00 | 2812.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-28 11:00:00 | 2850.90 | 2026-01-29 09:15:00 | 2780.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-01-28 11:45:00 | 2856.60 | 2026-01-29 09:15:00 | 2780.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2830.70 | 2026-02-04 09:15:00 | 2689.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 11:45:00 | 2830.50 | 2026-02-04 09:15:00 | 2688.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2830.70 | 2026-02-11 12:15:00 | 2547.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 11:45:00 | 2830.50 | 2026-02-11 12:15:00 | 2547.45 | TARGET_HIT | 0.50 | 10.00% |
