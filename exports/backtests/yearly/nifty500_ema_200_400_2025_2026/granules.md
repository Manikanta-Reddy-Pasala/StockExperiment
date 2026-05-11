# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 750.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 1 |
| TARGET_HIT | 6 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 22
- **Target hits / Stop hits / Partials:** 6 / 22 / 1
- **Avg / median % per leg:** 0.21% / -2.75%
- **Sum % (uncompounded):** 6.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 5 | 18.5% | 5 | 22 | 0 | -0.33% | -8.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 5 | 18.5% | 5 | 22 | 0 | -0.33% | -8.9% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 7 | 24.1% | 6 | 22 | 1 | 0.21% | 6.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 529.55 | 496.10 | 495.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 531.85 | 497.03 | 496.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 511.35 | 513.68 | 506.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 12:00:00 | 511.35 | 513.68 | 506.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 500.50 | 513.38 | 506.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 499.50 | 513.38 | 506.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 506.75 | 513.31 | 506.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 509.05 | 513.31 | 506.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 499.00 | 513.05 | 506.51 | SL hit (close<static) qty=1.00 sl=500.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 489.15 | 501.95 | 501.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 486.05 | 499.97 | 500.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 495.40 | 493.15 | 496.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 498.05 | 493.26 | 496.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 498.65 | 493.26 | 496.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 499.55 | 493.32 | 496.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 499.55 | 493.32 | 496.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 503.35 | 493.42 | 496.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 510.60 | 493.42 | 496.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 498.90 | 494.32 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 500.25 | 494.32 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 498.70 | 494.39 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 500.20 | 494.39 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 496.65 | 494.41 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 498.60 | 494.41 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 494.75 | 494.44 | 497.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 496.25 | 494.44 | 497.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 491.55 | 487.89 | 493.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 484.85 | 488.08 | 492.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:15:00 | 460.61 | 486.90 | 492.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-06 10:15:00 | 436.37 | 479.92 | 487.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 518.15 | 483.19 | 483.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 520.20 | 483.91 | 483.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 511.75 | 513.60 | 502.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 511.75 | 513.60 | 502.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 537.20 | 554.63 | 538.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 537.20 | 554.63 | 538.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 536.05 | 554.45 | 538.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 542.60 | 553.89 | 538.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:45:00 | 541.75 | 553.77 | 538.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 540.40 | 553.37 | 538.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 541.50 | 553.24 | 538.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 537.85 | 552.89 | 538.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 537.15 | 552.89 | 538.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 539.00 | 552.75 | 538.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 542.20 | 552.75 | 538.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 534.30 | 552.04 | 541.30 | SL hit (close<static) qty=1.00 sl=535.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 11:15:00 | 509.05 | 2025-06-17 12:15:00 | 499.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-31 09:15:00 | 484.85 | 2025-08-01 10:15:00 | 460.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 484.85 | 2025-08-06 10:15:00 | 436.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-11-12 09:15:00 | 542.60 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-11-12 09:45:00 | 541.75 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-12 12:45:00 | 540.40 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-12 13:45:00 | 541.50 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-13 11:15:00 | 542.20 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-25 11:15:00 | 541.75 | 2025-12-19 09:15:00 | 595.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-25 12:30:00 | 540.95 | 2025-12-19 09:15:00 | 595.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-08 14:45:00 | 542.55 | 2025-12-19 09:15:00 | 596.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-19 14:15:00 | 581.45 | 2026-01-20 10:15:00 | 564.10 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-01-23 09:45:00 | 581.90 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-01-23 10:15:00 | 584.25 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-01-23 12:15:00 | 581.45 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2026-01-27 09:45:00 | 583.40 | 2026-01-27 12:15:00 | 573.45 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-02-04 09:15:00 | 587.05 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2026-02-04 12:15:00 | 584.35 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-02-04 15:15:00 | 583.95 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2026-02-20 13:30:00 | 585.15 | 2026-03-02 12:15:00 | 569.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-06 12:30:00 | 585.35 | 2026-03-09 09:15:00 | 558.85 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2026-03-11 09:15:00 | 591.25 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-03-11 13:15:00 | 586.85 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-03-12 11:00:00 | 588.15 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-12 15:00:00 | 588.65 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-18 14:45:00 | 586.60 | 2026-03-19 12:15:00 | 570.45 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-20 09:15:00 | 588.15 | 2026-03-23 11:15:00 | 569.30 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-03-23 13:15:00 | 571.15 | 2026-03-25 11:15:00 | 628.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 13:45:00 | 572.00 | 2026-03-25 11:15:00 | 629.20 | TARGET_HIT | 1.00 | 10.00% |
