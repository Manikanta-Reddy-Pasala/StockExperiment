# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 13 |
| TARGET_HIT | 9 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 17
- **Target hits / Stop hits / Partials:** 5 / 25 / 13
- **Avg / median % per leg:** 2.68% / 3.40%
- **Sum % (uncompounded):** 115.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.66% | -5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.66% | -5.3% |
| SELL (all) | 41 | 26 | 63.4% | 5 | 23 | 13 | 2.94% | 120.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 26 | 63.4% | 5 | 23 | 13 | 2.94% | 120.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 26 | 60.5% | 5 | 25 | 13 | 2.68% | 115.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 503.85 | 531.83 | 531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 502.20 | 531.26 | 531.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 518.20 | 506.28 | 516.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 515.50 | 506.37 | 516.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 12:30:00 | 510.50 | 506.51 | 516.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:15:00 | 511.00 | 506.51 | 516.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 510.50 | 506.62 | 516.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 511.00 | 506.69 | 516.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 485.45 | 504.60 | 513.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 485.45 | 504.60 | 513.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 484.97 | 504.40 | 513.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 484.97 | 504.40 | 513.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 492.95 | 494.05 | 505.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 494.10 | 494.05 | 505.47 | SL hit (close>ema200) qty=0.50 sl=494.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 494.10 | 494.05 | 505.47 | SL hit (close>ema200) qty=0.50 sl=494.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 494.10 | 494.05 | 505.47 | SL hit (close>ema200) qty=0.50 sl=494.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 494.10 | 494.05 | 505.47 | SL hit (close>ema200) qty=0.50 sl=494.05 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 481.70 | 493.83 | 505.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 481.95 | 493.72 | 504.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:45:00 | 482.05 | 492.72 | 503.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 479.55 | 491.95 | 502.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 497.30 | 489.22 | 499.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 497.30 | 489.22 | 499.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 492.70 | 489.25 | 498.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 489.15 | 489.39 | 498.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 491.55 | 489.45 | 498.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:00:00 | 490.80 | 489.54 | 498.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 490.55 | 488.11 | 496.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 495.25 | 488.20 | 496.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 496.05 | 488.20 | 496.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 496.50 | 488.28 | 496.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 495.15 | 488.36 | 496.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 494.65 | 488.42 | 496.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 493.15 | 488.68 | 496.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=500.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=499.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=499.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 506.55 | 489.08 | 495.97 | SL hit (close>static) qty=1.00 sl=499.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 495.00 | 491.58 | 496.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 495.50 | 491.62 | 496.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 495.50 | 491.62 | 496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 495.40 | 491.66 | 496.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 496.05 | 491.66 | 496.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 496.95 | 491.71 | 496.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 499.45 | 491.79 | 496.50 | SL hit (close>static) qty=1.00 sl=499.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 493.35 | 491.85 | 496.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 493.45 | 491.85 | 496.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 491.80 | 491.86 | 496.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 493.95 | 491.88 | 496.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 495.00 | 492.05 | 496.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 496.00 | 492.05 | 496.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 495.00 | 492.08 | 496.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 493.05 | 492.15 | 496.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 493.05 | 492.16 | 496.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 491.00 | 492.18 | 496.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=499.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=499.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=499.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=499.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=497.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=497.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 500.00 | 492.25 | 496.15 | SL hit (close>static) qty=1.00 sl=497.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 493.05 | 492.43 | 496.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 468.40 | 488.91 | 493.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 457.61 | 479.98 | 487.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 457.85 | 479.98 | 487.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 457.95 | 479.98 | 487.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 455.57 | 479.98 | 487.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 433.53 | 462.71 | 472.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 433.75 | 462.71 | 472.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 433.85 | 462.71 | 472.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 431.60 | 462.71 | 472.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 443.75 | 462.71 | 472.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 361.65 | 343.45 | 361.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 361.65 | 343.45 | 361.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 360.15 | 343.61 | 361.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:15:00 | 364.00 | 343.61 | 361.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 364.15 | 343.82 | 361.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 357.75 | 344.46 | 361.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 359.30 | 344.74 | 361.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 15:15:00 | 341.33 | 345.31 | 360.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:15:00 | 339.86 | 345.17 | 360.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 345.60 | 344.79 | 359.54 | SL hit (close>ema200) qty=0.50 sl=344.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 345.60 | 344.79 | 359.54 | SL hit (close>ema200) qty=0.50 sl=344.79 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 14:45:00 | 358.95 | 336.40 | 347.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 10:15:00 | 341.00 | 338.54 | 347.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 10:15:00 | 341.30 | 338.54 | 347.32 | SL hit (close>static) qty=0.50 sl=338.54 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 359.65 | 337.56 | 340.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 341.67 | 340.32 | 341.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 342.70 | 340.32 | 341.98 | SL hit (close>static) qty=0.50 sl=340.32 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 344.30 | 340.36 | 341.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 343.90 | 340.36 | 341.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 343.30 | 340.95 | 342.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 342.95 | 340.95 | 342.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 344.25 | 340.98 | 342.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 345.55 | 340.98 | 342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 15:15:00 | 539.95 | 2025-06-18 13:15:00 | 526.50 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-06-16 10:45:00 | 541.85 | 2025-06-18 13:15:00 | 526.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-18 12:30:00 | 510.50 | 2025-07-25 10:15:00 | 485.45 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-07-18 13:15:00 | 511.00 | 2025-07-25 10:15:00 | 485.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 15:00:00 | 510.50 | 2025-07-25 11:15:00 | 484.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 10:00:00 | 511.00 | 2025-07-25 11:15:00 | 484.97 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-07-18 12:30:00 | 510.50 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-07-18 13:15:00 | 511.00 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-07-18 15:00:00 | 510.50 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-07-21 10:00:00 | 511.00 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-08-07 09:30:00 | 481.70 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.16% |
| SELL | retest2 | 2025-08-07 11:15:00 | 481.95 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2025-08-12 10:45:00 | 482.05 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-08-13 09:30:00 | 479.55 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2025-08-22 15:00:00 | 489.15 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-08-25 09:15:00 | 491.55 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-25 13:00:00 | 490.80 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-09-02 15:15:00 | 490.55 | 2025-09-12 10:15:00 | 499.45 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-03 12:15:00 | 495.15 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-03 12:45:00 | 494.65 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-04 11:00:00 | 493.15 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-11 13:30:00 | 495.00 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-12 13:15:00 | 493.35 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-12 14:45:00 | 493.45 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-15 09:45:00 | 491.80 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-09-26 09:15:00 | 468.40 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-09-17 13:00:00 | 493.05 | 2025-10-09 09:15:00 | 457.61 | PARTIAL | 0.50 | 7.19% |
| SELL | retest2 | 2025-09-17 14:00:00 | 493.05 | 2025-10-09 09:15:00 | 457.85 | PARTIAL | 0.50 | 7.14% |
| SELL | retest2 | 2025-09-17 15:15:00 | 491.00 | 2025-10-09 09:15:00 | 457.95 | PARTIAL | 0.50 | 6.73% |
| SELL | retest2 | 2025-09-18 15:00:00 | 493.05 | 2025-10-09 09:15:00 | 455.57 | PARTIAL | 0.50 | 7.60% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-11-06 09:15:00 | 433.53 | TARGET_HIT | 0.50 | 12.23% |
| SELL | retest2 | 2025-09-17 13:00:00 | 493.05 | 2025-11-06 09:15:00 | 433.75 | TARGET_HIT | 0.50 | 12.03% |
| SELL | retest2 | 2025-09-17 14:00:00 | 493.05 | 2025-11-06 09:15:00 | 433.85 | TARGET_HIT | 0.50 | 12.01% |
| SELL | retest2 | 2025-09-17 15:15:00 | 491.00 | 2025-11-06 09:15:00 | 431.60 | TARGET_HIT | 0.50 | 12.10% |
| SELL | retest2 | 2025-09-18 15:00:00 | 493.05 | 2025-11-06 09:15:00 | 443.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 357.75 | 2026-02-13 15:15:00 | 341.33 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-12 10:45:00 | 359.30 | 2026-02-16 11:15:00 | 339.86 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2026-02-12 09:15:00 | 357.75 | 2026-02-17 13:15:00 | 345.60 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2026-02-12 10:45:00 | 359.30 | 2026-02-17 13:15:00 | 345.60 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-03-16 14:45:00 | 358.95 | 2026-03-19 10:15:00 | 341.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 14:45:00 | 358.95 | 2026-03-19 10:15:00 | 341.30 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2026-04-27 14:45:00 | 359.65 | 2026-04-30 10:15:00 | 341.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:45:00 | 359.65 | 2026-04-30 10:15:00 | 342.70 | STOP_HIT | 0.50 | 4.71% |
