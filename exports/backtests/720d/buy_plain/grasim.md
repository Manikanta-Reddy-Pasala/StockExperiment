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
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 0 / 8 / 3
- **Avg / median % per leg:** 5.65% / 7.38%
- **Sum % (uncompounded):** 62.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 0 | 8 | 3 | 5.65% | 62.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 7 | 63.6% | 0 | 8 | 3 | 5.65% | 62.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 7 | 63.6% | 0 | 8 | 3 | 5.65% | 62.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 2605.50 | 2471.26 | 2471.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 2626.95 | 2472.80 | 2471.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.59 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.59 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 2581.50 | 2509.41 | 2492.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 2582.30 | 2510.14 | 2492.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 2520.30 | 2661.08 | 2630.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 2523.50 | 2659.71 | 2630.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-03 09:15:00 | 2536.00 | 2654.53 | 2628.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 2547.00 | 2653.46 | 2627.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-22 10:15:00 | 2902.02 | 2810.35 | 2782.58 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 2813.00 | 2813.64 | 2785.35 | SL hit (close<ema200) qty=0.50 sl=2813.64 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-27 15:15:00 | 2929.05 | 2823.07 | 2803.06 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-29 09:15:00 | 2969.64 | 2831.42 | 2808.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2735.00 | 2855.01 | 2824.94 | SL hit (close<ema200) qty=0.50 sl=2855.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2735.00 | 2855.01 | 2824.94 | SL hit (close<ema200) qty=0.50 sl=2855.01 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2787.98 | 2787.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.03 | 2790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.82 | 2804.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 2827.40 | 2796.19 | 2796.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 2813.30 | 2796.36 | 2796.53 | ENTRY2 sustain failed after 60m |

### Cycle 3 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.82 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2857.90 | 2798.38 | 2797.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2835.80 | 2800.40 | 2799.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-04 09:15:00 | 2852.50 | 2801.57 | 2799.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 2842.50 | 2801.98 | 2799.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 13:15:00 | 2818.60 | 2858.53 | 2838.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-27 14:15:00 | 2796.00 | 2857.91 | 2838.34 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2752.90 | 2856.23 | 2837.69 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2752.90 | 2856.23 | 2837.69 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 2824.40 | 2720.23 | 2735.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 2833.00 | 2721.35 | 2735.67 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-28 14:15:00 | 2784.10 | 2724.67 | 2737.06 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-29 11:15:00 | 2831.90 | 2727.93 | 2738.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:15:00 | 2820.00 | 2728.85 | 2738.87 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 2749.00 | 2731.15 | 2739.83 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 2749.00 | 2731.15 | 2739.83 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 13:15:00 | 2788.90 | 2732.69 | 2740.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 2798.70 | 2733.35 | 2740.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 2875.50 | 2748.38 | 2747.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 2875.50 | 2748.38 | 2747.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.00 | 2750.88 | 2749.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-08 10:15:00 | 2582.30 | 2025-09-22 10:15:00 | 2902.02 | PARTIAL | 0.50 | 12.38% |
| BUY | retest2 | 2025-04-08 10:15:00 | 2582.30 | 2025-09-23 11:15:00 | 2813.00 | STOP_HIT | 0.50 | 8.93% |
| BUY | retest2 | 2025-06-02 12:15:00 | 2523.50 | 2025-10-27 15:15:00 | 2929.05 | PARTIAL | 0.50 | 16.07% |
| BUY | retest2 | 2025-06-03 10:15:00 | 2547.00 | 2025-10-29 09:15:00 | 2969.64 | PARTIAL | 0.50 | 16.59% |
| BUY | retest2 | 2025-06-02 12:15:00 | 2523.50 | 2025-11-06 09:15:00 | 2735.00 | STOP_HIT | 0.50 | 8.38% |
| BUY | retest2 | 2025-06-03 10:15:00 | 2547.00 | 2025-11-06 09:15:00 | 2735.00 | STOP_HIT | 0.50 | 7.38% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2835.80 | 2026-03-02 09:15:00 | 2752.90 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-02-04 10:15:00 | 2842.50 | 2026-03-02 09:15:00 | 2752.90 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2026-04-28 10:15:00 | 2833.00 | 2026-04-28 14:15:00 | 2784.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-29 12:15:00 | 2820.00 | 2026-04-30 09:15:00 | 2749.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-04-30 14:15:00 | 2798.70 | 2026-05-05 14:15:00 | 2875.50 | STOP_HIT | 1.00 | 2.74% |
