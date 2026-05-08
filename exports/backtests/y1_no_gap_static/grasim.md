# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.44% / -1.88%
- **Sum % (uncompounded):** -9.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.04% | -6.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.04% | -6.1% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.84% | -3.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.84% | -3.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.44% | -9.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.65 | 2803.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 2723.00 | 2800.53 | 2802.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.39 | 2778.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.29 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-09 13:15:00 | 2761.30 | 2762.45 | 2778.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 2748.30 | 2762.31 | 2778.06 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 12:15:00 | 2748.20 | 2762.17 | 2777.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 2746.40 | 2762.02 | 2777.44 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.16 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.16 | SL hit (close>static) qty=1.00 sl=2779.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2787.98 | 2787.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.03 | 2790.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.82 | 2804.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.43 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.54 | 2796.61 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.82 | 2796.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2857.90 | 2798.38 | 2797.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.34 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2835.80 | 2800.40 | 2799.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-04 09:15:00 | 2852.50 | 2801.57 | 2799.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 2842.50 | 2801.98 | 2799.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 13:15:00 | 2818.60 | 2858.53 | 2838.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-27 14:15:00 | 2796.00 | 2857.91 | 2838.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2752.90 | 2856.23 | 2837.68 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2752.90 | 2856.23 | 2837.68 | SL hit (close<static) qty=1.00 sl=2787.90 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2630.00 | 2793.23 | 2806.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 2875.50 | 2748.38 | 2747.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.00 | 2750.88 | 2749.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-09 14:15:00 | 2748.30 | 2025-12-11 13:15:00 | 2797.90 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-12-10 13:15:00 | 2746.40 | 2025-12-11 13:15:00 | 2797.90 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2835.80 | 2026-03-02 09:15:00 | 2752.90 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-02-04 10:15:00 | 2842.50 | 2026-03-02 09:15:00 | 2752.90 | STOP_HIT | 1.00 | -3.15% |
