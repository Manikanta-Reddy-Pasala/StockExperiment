# Elgi Equipments Ltd. (ELGIEQUIP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 561.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 13 |
| TARGET_HIT | 3 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 16
- **Target hits / Stop hits / Partials:** 3 / 26 / 13
- **Avg / median % per leg:** 1.98% / 2.59%
- **Sum % (uncompounded):** 83.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.79% | -11.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.79% | -11.2% |
| SELL (all) | 38 | 26 | 68.4% | 3 | 22 | 13 | 2.48% | 94.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 26 | 68.4% | 3 | 22 | 13 | 2.48% | 94.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 26 | 61.9% | 3 | 26 | 13 | 1.98% | 83.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 535.00 | 480.58 | 480.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 536.35 | 481.13 | 480.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 508.65 | 510.42 | 499.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 508.65 | 510.42 | 499.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 543.10 | 552.93 | 537.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 495.65 | 527.03 | 527.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 489.10 | 526.36 | 526.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 496.35 | 492.38 | 505.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 496.35 | 492.38 | 505.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 511.85 | 492.64 | 505.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 511.85 | 492.64 | 505.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 506.60 | 492.77 | 505.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 505.05 | 492.90 | 505.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 503.55 | 493.01 | 505.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 503.95 | 493.37 | 505.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 505.10 | 494.05 | 505.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 499.40 | 494.11 | 505.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:00:00 | 496.80 | 494.26 | 505.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 496.60 | 494.33 | 505.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 497.55 | 494.54 | 504.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 498.30 | 494.64 | 504.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 504.25 | 494.73 | 504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 504.25 | 494.73 | 504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 501.30 | 494.80 | 504.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 498.10 | 495.02 | 504.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 479.80 | 494.86 | 504.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 479.85 | 494.86 | 504.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 478.37 | 494.70 | 504.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 478.75 | 494.70 | 504.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 473.38 | 493.76 | 503.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 473.19 | 493.76 | 503.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 497.50 | 491.67 | 500.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 471.96 | 490.26 | 498.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 471.77 | 490.26 | 498.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 472.67 | 490.26 | 498.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 472.62 | 490.26 | 498.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 482.75 | 481.47 | 491.68 | SL hit (close>ema200) qty=0.50 sl=481.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 482.75 | 481.47 | 491.68 | SL hit (close>ema200) qty=0.50 sl=481.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 482.75 | 481.47 | 491.68 | SL hit (close>ema200) qty=0.50 sl=481.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 482.75 | 481.47 | 491.68 | SL hit (close>ema200) qty=0.50 sl=481.47 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:30:00 | 499.85 | 483.49 | 491.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 505.30 | 484.10 | 491.99 | SL hit (close>static) qty=1.00 sl=505.05 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 499.80 | 484.46 | 492.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 507.45 | 485.60 | 492.41 | SL hit (close>static) qty=1.00 sl=505.05 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 490.75 | 487.13 | 492.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 488.55 | 487.13 | 492.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 489.40 | 485.09 | 490.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:45:00 | 489.35 | 485.21 | 490.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 488.65 | 485.26 | 490.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 490.70 | 485.38 | 490.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 490.70 | 485.38 | 490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 492.25 | 485.44 | 490.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 492.25 | 485.44 | 490.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 490.00 | 485.49 | 490.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 496.80 | 485.49 | 490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 499.25 | 485.63 | 490.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 499.25 | 485.63 | 490.81 | SL hit (close>static) qty=1.00 sl=494.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 499.25 | 485.63 | 490.81 | SL hit (close>static) qty=1.00 sl=494.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 499.25 | 485.63 | 490.81 | SL hit (close>static) qty=1.00 sl=494.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 499.25 | 485.63 | 490.81 | SL hit (close>static) qty=1.00 sl=494.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 505.05 | 485.63 | 490.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 494.00 | 485.71 | 490.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 498.40 | 485.71 | 490.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 495.40 | 492.56 | 493.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 15:00:00 | 492.25 | 492.61 | 493.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 492.00 | 492.62 | 493.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 491.95 | 492.62 | 493.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 493.05 | 492.62 | 493.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 492.20 | 492.62 | 493.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 493.15 | 492.62 | 493.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 493.60 | 492.63 | 493.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 493.60 | 492.63 | 493.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 498.40 | 492.65 | 493.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 498.40 | 492.65 | 493.64 | SL hit (close>static) qty=1.00 sl=497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 498.40 | 492.65 | 493.64 | SL hit (close>static) qty=1.00 sl=497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 498.40 | 492.65 | 493.64 | SL hit (close>static) qty=1.00 sl=497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 498.40 | 492.65 | 493.64 | SL hit (close>static) qty=1.00 sl=497.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 498.40 | 492.65 | 493.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 491.80 | 492.64 | 493.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 489.00 | 492.65 | 493.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 477.15 | 492.54 | 493.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 504.05 | 492.05 | 493.29 | SL hit (close>static) qty=1.00 sl=498.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 504.05 | 492.05 | 493.29 | SL hit (close>static) qty=1.00 sl=498.45 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 508.10 | 494.41 | 494.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 509.95 | 494.99 | 494.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 496.70 | 496.76 | 495.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:45:00 | 496.40 | 496.76 | 495.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 493.20 | 496.72 | 495.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 493.20 | 496.72 | 495.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 491.50 | 496.67 | 495.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 497.00 | 496.67 | 495.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 493.70 | 496.80 | 495.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 487.50 | 496.70 | 495.67 | SL hit (close<static) qty=1.00 sl=489.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 487.50 | 496.70 | 495.67 | SL hit (close<static) qty=1.00 sl=489.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:00:00 | 494.20 | 495.72 | 495.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 484.70 | 495.50 | 495.12 | SL hit (close<static) qty=1.00 sl=489.85 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 484.35 | 494.74 | 494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 483.75 | 494.63 | 494.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 483.25 | 480.27 | 485.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:45:00 | 483.10 | 480.27 | 485.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 481.10 | 480.43 | 485.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 477.75 | 480.36 | 485.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 476.90 | 480.36 | 485.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 15:00:00 | 477.75 | 480.33 | 485.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.86 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.05 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.86 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 15:15:00 | 429.98 | 466.64 | 476.75 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 15:15:00 | 429.21 | 466.64 | 476.75 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 15:15:00 | 429.98 | 466.64 | 476.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 515.00 | 472.01 | 471.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 516.95 | 475.95 | 473.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 501.30 | 504.19 | 490.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 501.30 | 504.19 | 490.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 500.75 | 505.39 | 492.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 516.35 | 505.37 | 493.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 485.00 | 506.23 | 494.55 | SL hit (close<static) qty=1.00 sl=492.55 alert=retest2 |

### Cycle 6 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 476.85 | 487.29 | 487.33 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 15:15:00 | 500.50 | 487.45 | 487.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 502.35 | 487.60 | 487.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-18 11:30:00 | 505.05 | 2025-09-26 10:15:00 | 479.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 12:30:00 | 503.55 | 2025-09-26 10:15:00 | 479.85 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-19 09:15:00 | 503.95 | 2025-09-26 11:15:00 | 478.37 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-09-22 09:30:00 | 505.10 | 2025-09-26 11:15:00 | 478.75 | PARTIAL | 0.50 | 5.22% |
| SELL | retest2 | 2025-09-22 14:00:00 | 496.80 | 2025-09-29 14:15:00 | 473.38 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-23 09:15:00 | 496.60 | 2025-09-29 14:15:00 | 473.19 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-18 11:30:00 | 505.05 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2025-09-18 12:30:00 | 503.55 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-09-19 09:15:00 | 503.95 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-09-22 09:30:00 | 505.10 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-09-22 14:00:00 | 496.80 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2025-09-23 09:15:00 | 496.60 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2025-09-24 09:15:00 | 497.55 | 2025-10-13 10:15:00 | 471.96 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-09-24 10:30:00 | 498.30 | 2025-10-13 10:15:00 | 471.77 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-09-25 09:30:00 | 498.10 | 2025-10-13 10:15:00 | 472.67 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-07 11:15:00 | 497.50 | 2025-10-13 10:15:00 | 472.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 497.55 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2025-09-24 10:30:00 | 498.30 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2025-09-25 09:30:00 | 498.10 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-10-07 11:15:00 | 497.50 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-10-29 11:30:00 | 499.85 | 2025-10-29 14:15:00 | 505.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-30 10:00:00 | 499.80 | 2025-10-31 09:15:00 | 507.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-04 11:15:00 | 488.55 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-11-11 11:00:00 | 489.40 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-12 09:45:00 | 489.35 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-11-12 10:45:00 | 488.65 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-11-21 15:00:00 | 492.25 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-24 09:30:00 | 492.00 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-11-24 10:15:00 | 491.95 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-24 10:45:00 | 493.05 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-25 09:30:00 | 489.00 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-25 10:45:00 | 477.15 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2025-12-05 09:15:00 | 497.00 | 2025-12-08 10:15:00 | 487.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-08 10:15:00 | 493.70 | 2025-12-08 10:15:00 | 487.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-09 15:00:00 | 494.20 | 2025-12-10 10:15:00 | 484.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-01-05 13:30:00 | 477.75 | 2026-01-09 09:15:00 | 453.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 476.90 | 2026-01-09 09:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 477.75 | 2026-01-09 09:15:00 | 453.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 477.75 | 2026-01-16 15:15:00 | 429.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 476.90 | 2026-01-16 15:15:00 | 429.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 477.75 | 2026-01-16 15:15:00 | 429.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-10 09:15:00 | 516.35 | 2026-03-12 09:15:00 | 485.00 | STOP_HIT | 1.00 | -6.07% |
