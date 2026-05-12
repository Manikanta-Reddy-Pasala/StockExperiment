# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-11 15:15:00 (3451 bars)
- **Last close:** 542.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 167 |
| ALERT1 | 105 |
| ALERT2 | 101 |
| ALERT2_SKIP | 56 |
| ALERT3 | 298 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 165 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 162 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 128
- **Target hits / Stop hits / Partials:** 4 / 162 / 12
- **Avg / median % per leg:** 0.18% / -0.74%
- **Sum % (uncompounded):** 32.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 25 | 34.7% | 4 | 68 | 0 | 0.56% | 40.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.78% | -1.8% |
| BUY @ 3rd Alert (retest2) | 71 | 25 | 35.2% | 4 | 67 | 0 | 0.59% | 41.9% |
| SELL (all) | 106 | 25 | 23.6% | 0 | 94 | 12 | -0.07% | -7.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 106 | 25 | 23.6% | 0 | 94 | 12 | -0.07% | -7.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.78% | -1.8% |
| retest2 (combined) | 177 | 50 | 28.2% | 4 | 161 | 12 | 0.19% | 34.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 572.85 | 569.69 | 569.37 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 10:15:00 | 563.45 | 568.49 | 568.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 562.45 | 567.28 | 568.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 536.50 | 531.68 | 539.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 536.50 | 531.68 | 539.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 536.50 | 531.68 | 539.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 541.30 | 531.68 | 539.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 540.10 | 533.93 | 539.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 540.25 | 533.93 | 539.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 536.35 | 534.42 | 538.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:30:00 | 543.15 | 534.42 | 538.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 535.95 | 534.72 | 538.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:45:00 | 533.00 | 534.80 | 538.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 15:15:00 | 532.50 | 534.80 | 538.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 09:45:00 | 533.55 | 534.40 | 537.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 532.40 | 534.40 | 537.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 524.25 | 529.65 | 533.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 13:45:00 | 522.00 | 526.20 | 530.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 519.50 | 524.25 | 528.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 09:15:00 | 506.35 | 518.47 | 522.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 524.45 | 518.47 | 522.48 | SL hit (close>static) qty=0.50 sl=518.47 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 529.05 | 524.98 | 524.71 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 506.00 | 521.74 | 523.36 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 523.75 | 520.12 | 520.07 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 517.00 | 519.50 | 519.79 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 530.10 | 521.62 | 520.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 537.15 | 527.24 | 523.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 514.25 | 526.14 | 524.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 514.25 | 526.14 | 524.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 514.25 | 526.14 | 524.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 508.95 | 526.14 | 524.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 476.35 | 516.18 | 520.16 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 527.05 | 511.65 | 511.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 15:15:00 | 527.30 | 514.78 | 512.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 12:15:00 | 520.00 | 520.08 | 516.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 13:00:00 | 520.00 | 520.08 | 516.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 586.20 | 591.87 | 585.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 586.20 | 591.87 | 585.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 583.00 | 590.10 | 585.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 585.35 | 590.10 | 585.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 581.75 | 588.43 | 585.16 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 15:15:00 | 579.00 | 583.46 | 583.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 574.85 | 581.74 | 582.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 10:15:00 | 582.10 | 581.81 | 582.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 10:15:00 | 582.10 | 581.81 | 582.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 582.10 | 581.81 | 582.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 582.10 | 581.81 | 582.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 582.30 | 581.91 | 582.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 12:15:00 | 579.60 | 581.91 | 582.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 09:15:00 | 583.70 | 570.30 | 569.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 583.70 | 570.30 | 569.12 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 570.50 | 572.19 | 572.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 15:15:00 | 568.40 | 571.43 | 571.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 561.00 | 557.47 | 560.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 561.00 | 557.47 | 560.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 561.00 | 557.47 | 560.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:00:00 | 552.50 | 556.27 | 559.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 10:45:00 | 554.50 | 554.91 | 557.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 554.10 | 554.38 | 556.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:45:00 | 553.60 | 553.14 | 555.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 550.60 | 552.63 | 554.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:15:00 | 549.85 | 552.24 | 554.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 13:30:00 | 550.00 | 551.38 | 553.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:15:00 | 549.90 | 544.73 | 547.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:45:00 | 549.90 | 545.80 | 547.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 13:15:00 | 555.30 | 549.61 | 549.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 555.30 | 549.61 | 549.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 556.60 | 551.01 | 549.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 549.75 | 551.72 | 550.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 549.75 | 551.72 | 550.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 549.75 | 551.72 | 550.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 549.75 | 551.72 | 550.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 553.00 | 551.97 | 550.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 555.50 | 552.82 | 551.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:00:00 | 555.10 | 552.82 | 551.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:45:00 | 554.85 | 553.49 | 551.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 555.80 | 553.34 | 552.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 558.80 | 558.16 | 555.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:30:00 | 557.15 | 558.16 | 555.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 556.60 | 557.85 | 555.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 551.55 | 557.85 | 555.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 567.85 | 559.85 | 556.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 569.40 | 559.85 | 556.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:45:00 | 569.40 | 563.32 | 559.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 571.05 | 563.32 | 559.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 13:15:00 | 575.25 | 578.14 | 578.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 575.25 | 578.14 | 578.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 572.30 | 576.33 | 577.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 582.20 | 577.51 | 577.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 582.20 | 577.51 | 577.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 582.20 | 577.51 | 577.94 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 11:15:00 | 580.55 | 578.31 | 578.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 12:15:00 | 583.70 | 579.39 | 578.74 | Break + close above crossover candle high |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 571.10 | 578.83 | 578.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 568.15 | 576.69 | 577.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 579.30 | 577.21 | 578.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 11:15:00 | 579.30 | 577.21 | 578.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 579.30 | 577.21 | 578.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:30:00 | 579.50 | 577.21 | 578.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 583.75 | 578.52 | 578.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 587.00 | 578.52 | 578.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 571.20 | 576.81 | 577.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 568.20 | 573.32 | 575.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 539.79 | 564.24 | 569.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 565.50 | 563.78 | 568.69 | SL hit (close>ema200) qty=0.50 sl=563.78 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 582.35 | 572.20 | 571.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 596.75 | 579.36 | 574.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 629.50 | 630.62 | 622.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 629.50 | 630.62 | 622.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 620.50 | 627.11 | 622.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 620.50 | 627.11 | 622.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 619.80 | 625.65 | 622.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 616.45 | 625.65 | 622.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 614.10 | 622.26 | 621.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 614.10 | 622.26 | 621.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 614.45 | 620.70 | 620.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 15:15:00 | 608.40 | 615.53 | 618.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 616.70 | 615.77 | 617.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 616.70 | 615.77 | 617.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 616.70 | 615.77 | 617.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 616.70 | 615.77 | 617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 561.20 | 562.99 | 570.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:15:00 | 558.25 | 562.99 | 570.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 575.70 | 563.90 | 567.97 | SL hit (close>static) qty=1.00 sl=572.70 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 572.15 | 569.92 | 569.62 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 565.80 | 570.86 | 570.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 558.80 | 566.44 | 568.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 543.50 | 541.67 | 550.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:30:00 | 546.45 | 541.67 | 550.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 550.05 | 545.36 | 549.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 551.10 | 545.36 | 549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 552.05 | 546.70 | 549.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 552.50 | 546.70 | 549.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 566.25 | 551.54 | 551.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 567.20 | 556.54 | 553.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 583.75 | 587.21 | 580.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 583.75 | 587.21 | 580.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 581.45 | 586.06 | 580.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 581.45 | 586.06 | 580.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 581.10 | 585.07 | 580.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:30:00 | 582.05 | 585.07 | 580.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 578.45 | 583.74 | 580.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 578.45 | 583.74 | 580.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 576.70 | 582.33 | 580.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 576.70 | 582.33 | 580.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 571.10 | 577.99 | 578.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 569.00 | 572.39 | 575.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 564.50 | 564.23 | 568.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:45:00 | 565.35 | 564.23 | 568.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 567.20 | 563.72 | 565.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:45:00 | 566.75 | 563.72 | 565.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 563.55 | 563.68 | 565.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 560.00 | 563.53 | 564.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 579.25 | 566.11 | 565.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 579.25 | 566.11 | 565.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 592.45 | 581.43 | 575.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 581.80 | 583.23 | 578.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 581.80 | 583.23 | 578.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 587.00 | 586.79 | 583.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 587.00 | 586.79 | 583.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 585.00 | 586.43 | 583.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 593.00 | 587.84 | 585.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 605.70 | 588.60 | 586.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-06 09:15:00 | 652.30 | 610.89 | 599.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 678.00 | 688.06 | 689.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 660.60 | 682.57 | 686.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 682.70 | 672.76 | 676.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 682.70 | 672.76 | 676.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 682.70 | 672.76 | 676.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:30:00 | 678.95 | 672.76 | 676.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 679.00 | 674.01 | 676.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 673.25 | 674.01 | 676.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 682.95 | 676.85 | 677.18 | SL hit (close>static) qty=1.00 sl=682.85 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 680.85 | 662.19 | 660.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 688.95 | 667.54 | 663.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 684.15 | 684.61 | 676.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:45:00 | 685.50 | 684.61 | 676.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 680.30 | 684.18 | 679.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 677.50 | 684.18 | 679.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 673.20 | 681.98 | 678.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 673.20 | 681.98 | 678.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 681.15 | 681.81 | 678.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 682.50 | 681.81 | 678.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 667.60 | 676.25 | 677.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 667.60 | 676.25 | 677.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 663.05 | 670.52 | 674.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 676.55 | 668.84 | 671.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 676.55 | 668.84 | 671.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 676.55 | 668.84 | 671.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 676.55 | 668.84 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 665.60 | 668.19 | 671.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 664.80 | 667.33 | 670.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 663.50 | 666.54 | 669.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 13:45:00 | 664.95 | 666.16 | 669.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 677.00 | 668.33 | 670.14 | SL hit (close>static) qty=1.00 sl=676.80 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 689.65 | 673.60 | 672.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 12:15:00 | 697.40 | 683.13 | 677.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 714.45 | 719.66 | 705.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 709.25 | 717.17 | 706.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 709.25 | 717.17 | 706.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 709.25 | 717.17 | 706.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 706.85 | 715.10 | 706.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 706.85 | 715.10 | 706.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 699.80 | 712.04 | 705.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 699.80 | 712.04 | 705.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 705.10 | 710.65 | 705.90 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 11:15:00 | 701.05 | 703.34 | 703.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 685.45 | 697.01 | 700.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 663.40 | 658.39 | 673.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 664.00 | 658.39 | 673.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 666.50 | 662.40 | 670.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 666.50 | 662.40 | 670.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 673.80 | 664.68 | 670.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 673.80 | 664.68 | 670.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 668.10 | 665.36 | 670.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 674.25 | 665.36 | 670.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 684.15 | 669.12 | 671.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 684.15 | 669.12 | 671.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 685.00 | 672.30 | 673.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 687.10 | 672.30 | 673.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 682.40 | 674.32 | 673.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 11:15:00 | 689.30 | 682.81 | 679.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 10:15:00 | 685.25 | 686.40 | 683.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 11:00:00 | 685.25 | 686.40 | 683.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 689.55 | 692.12 | 688.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 690.00 | 692.12 | 688.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 690.90 | 692.43 | 690.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 701.00 | 691.03 | 690.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 683.25 | 689.31 | 689.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 683.25 | 689.31 | 689.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 680.00 | 686.49 | 688.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 691.15 | 685.78 | 687.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 691.15 | 685.78 | 687.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 691.15 | 685.78 | 687.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:45:00 | 690.65 | 685.78 | 687.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 698.05 | 688.24 | 688.11 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 670.20 | 685.92 | 687.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 662.10 | 681.15 | 685.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 669.45 | 667.02 | 675.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 669.45 | 667.02 | 675.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 666.05 | 656.89 | 665.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:30:00 | 659.90 | 656.89 | 665.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 665.60 | 658.63 | 665.15 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 14:15:00 | 674.70 | 669.34 | 668.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 12:15:00 | 682.45 | 673.60 | 671.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 672.70 | 677.55 | 674.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 672.70 | 677.55 | 674.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 672.70 | 677.55 | 674.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 672.70 | 677.55 | 674.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 667.60 | 675.56 | 673.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:00:00 | 667.60 | 675.56 | 673.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 673.95 | 675.24 | 673.70 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 663.15 | 671.73 | 672.30 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 09:15:00 | 682.00 | 674.17 | 673.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 10:15:00 | 689.30 | 677.19 | 674.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 643.00 | 677.14 | 677.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 643.00 | 677.14 | 677.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 643.00 | 677.14 | 677.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 643.00 | 677.14 | 677.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 644.55 | 670.62 | 674.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 630.00 | 646.25 | 656.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 636.90 | 629.84 | 638.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 636.90 | 629.84 | 638.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 636.90 | 629.84 | 638.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 636.90 | 629.84 | 638.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 636.30 | 631.13 | 638.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 625.65 | 631.13 | 638.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 641.50 | 632.49 | 636.56 | SL hit (close>static) qty=1.00 sl=639.45 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 655.25 | 641.83 | 640.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 10:15:00 | 670.45 | 650.14 | 644.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 12:15:00 | 698.95 | 702.13 | 690.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 698.95 | 702.13 | 690.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 690.10 | 698.78 | 691.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 692.95 | 698.78 | 691.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 687.95 | 696.61 | 690.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 685.90 | 696.61 | 690.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 684.50 | 692.60 | 689.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:15:00 | 680.00 | 692.60 | 689.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 678.60 | 687.21 | 687.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 672.10 | 684.19 | 686.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 666.50 | 657.44 | 664.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 666.50 | 657.44 | 664.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 666.50 | 657.44 | 664.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 666.50 | 657.44 | 664.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 659.75 | 657.90 | 664.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 648.60 | 660.96 | 663.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 657.65 | 658.64 | 661.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:45:00 | 658.60 | 658.43 | 661.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 657.05 | 657.31 | 659.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 647.20 | 655.29 | 658.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 644.80 | 650.24 | 655.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 660.35 | 643.89 | 643.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 660.35 | 643.89 | 643.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 667.50 | 648.61 | 645.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 663.90 | 665.16 | 659.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 14:00:00 | 663.90 | 665.16 | 659.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 668.55 | 666.61 | 662.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 663.45 | 666.61 | 662.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 676.20 | 680.11 | 675.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:00:00 | 682.20 | 679.00 | 675.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 15:00:00 | 683.90 | 679.98 | 676.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:00:00 | 683.35 | 680.95 | 677.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 671.05 | 678.78 | 677.42 | SL hit (close<static) qty=1.00 sl=675.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 14:15:00 | 667.00 | 674.75 | 675.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 627.05 | 663.83 | 670.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 10:15:00 | 636.70 | 635.20 | 647.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 636.70 | 635.20 | 647.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 594.00 | 592.14 | 597.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 592.70 | 592.14 | 597.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 591.10 | 591.93 | 596.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 586.85 | 594.22 | 596.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:00:00 | 589.85 | 592.64 | 594.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:00:00 | 589.95 | 592.10 | 594.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:45:00 | 589.65 | 591.63 | 594.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 587.90 | 590.17 | 592.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 584.60 | 588.31 | 591.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 585.30 | 587.75 | 590.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 584.80 | 587.30 | 588.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:45:00 | 585.00 | 587.60 | 588.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 588.00 | 587.68 | 588.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 594.80 | 587.68 | 588.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 596.25 | 589.39 | 589.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 596.25 | 589.39 | 589.45 | SL hit (close>static) qty=1.00 sl=595.10 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 602.80 | 592.07 | 590.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 12:15:00 | 606.90 | 596.55 | 593.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 606.50 | 607.72 | 602.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 606.50 | 607.72 | 602.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 603.90 | 606.86 | 602.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 603.90 | 606.86 | 602.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 598.90 | 605.92 | 603.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 598.90 | 605.92 | 603.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 592.00 | 603.13 | 602.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 588.10 | 603.13 | 602.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 585.65 | 599.64 | 600.85 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 601.50 | 594.44 | 594.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 606.10 | 601.20 | 598.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 602.40 | 602.73 | 599.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 13:00:00 | 602.40 | 602.73 | 599.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 601.80 | 602.34 | 600.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:15:00 | 600.00 | 602.34 | 600.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 600.00 | 601.87 | 600.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:30:00 | 602.00 | 601.28 | 600.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:00:00 | 601.85 | 601.39 | 600.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 603.55 | 601.42 | 600.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 603.00 | 602.80 | 601.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 602.50 | 604.88 | 603.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 604.20 | 604.88 | 603.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 602.95 | 604.50 | 603.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 603.85 | 604.50 | 603.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 602.90 | 604.18 | 603.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:00:00 | 604.95 | 604.10 | 603.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 606.15 | 603.45 | 603.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 598.55 | 602.53 | 602.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 598.55 | 602.53 | 602.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 596.00 | 600.66 | 602.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 600.55 | 600.40 | 601.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 600.55 | 600.40 | 601.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 600.55 | 600.40 | 601.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 600.05 | 600.40 | 601.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 595.00 | 599.32 | 601.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:30:00 | 598.60 | 599.32 | 601.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 609.95 | 598.34 | 599.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 609.10 | 598.34 | 599.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 616.80 | 602.04 | 600.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 620.20 | 608.38 | 604.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 12:15:00 | 618.85 | 619.12 | 613.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 12:45:00 | 618.95 | 619.12 | 613.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 622.90 | 628.33 | 624.15 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 613.50 | 621.71 | 622.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 606.70 | 617.71 | 620.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 618.90 | 608.92 | 613.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 618.90 | 608.92 | 613.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 618.90 | 608.92 | 613.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 618.90 | 608.92 | 613.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 616.65 | 610.46 | 613.39 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 626.65 | 616.64 | 615.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 14:15:00 | 631.95 | 619.70 | 617.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 09:15:00 | 641.75 | 643.17 | 634.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 09:30:00 | 639.10 | 643.17 | 634.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 639.50 | 640.40 | 636.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 630.00 | 640.40 | 636.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 628.75 | 638.07 | 635.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 627.10 | 638.07 | 635.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 642.30 | 638.92 | 636.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 647.10 | 641.06 | 637.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:45:00 | 648.20 | 645.14 | 640.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 635.95 | 655.28 | 657.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 635.95 | 655.28 | 657.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 628.25 | 649.88 | 655.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 617.10 | 609.52 | 621.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 617.10 | 609.52 | 621.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 617.10 | 609.52 | 621.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 618.90 | 609.52 | 621.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 621.75 | 611.97 | 621.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 621.75 | 611.97 | 621.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 610.55 | 611.69 | 620.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:30:00 | 609.55 | 611.42 | 619.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 626.00 | 615.46 | 619.88 | SL hit (close>static) qty=1.00 sl=622.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 11:15:00 | 610.05 | 607.78 | 607.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 12:15:00 | 617.15 | 609.65 | 608.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 15:15:00 | 618.00 | 619.21 | 616.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:15:00 | 612.60 | 619.21 | 616.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 608.95 | 617.15 | 615.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 608.95 | 617.15 | 615.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 608.95 | 615.51 | 614.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 608.20 | 615.51 | 614.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 605.70 | 613.55 | 613.99 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 625.80 | 613.97 | 613.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 10:15:00 | 631.00 | 622.59 | 618.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 15:15:00 | 626.65 | 628.58 | 623.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 09:15:00 | 634.30 | 628.58 | 623.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 626.40 | 629.27 | 624.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 626.40 | 629.27 | 624.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 623.00 | 628.01 | 624.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 623.00 | 628.01 | 624.69 | SL hit (close<ema400) qty=1.00 sl=624.69 alert=retest1 |

### Cycle 52 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 616.05 | 621.89 | 622.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 598.15 | 617.14 | 620.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 596.20 | 596.14 | 604.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 597.55 | 596.14 | 604.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 599.80 | 596.35 | 601.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 593.80 | 599.12 | 600.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 590.25 | 599.12 | 600.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 587.90 | 583.63 | 590.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 564.11 | 577.29 | 584.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 574.00 | 569.95 | 576.06 | SL hit (close>ema200) qty=0.50 sl=569.95 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 528.00 | 517.46 | 516.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 547.85 | 530.76 | 527.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 533.30 | 533.50 | 530.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 533.30 | 533.50 | 530.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 519.40 | 530.76 | 529.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 516.10 | 530.76 | 529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 523.95 | 529.40 | 528.91 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 517.35 | 526.99 | 527.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 513.15 | 524.22 | 526.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 524.55 | 517.85 | 522.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 524.55 | 517.85 | 522.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 524.55 | 517.85 | 522.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:45:00 | 529.20 | 517.85 | 522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 544.35 | 523.15 | 524.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 544.35 | 523.15 | 524.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 11:15:00 | 542.65 | 527.05 | 525.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 14:15:00 | 573.45 | 541.81 | 533.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 537.45 | 545.90 | 536.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 537.45 | 545.90 | 536.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 537.45 | 545.90 | 536.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 535.00 | 545.90 | 536.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 531.65 | 543.05 | 536.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 531.65 | 543.05 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 525.80 | 539.60 | 535.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 525.20 | 539.60 | 535.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 521.00 | 532.03 | 532.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 13:15:00 | 518.75 | 523.67 | 527.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 521.20 | 520.14 | 524.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 521.20 | 520.14 | 524.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 521.20 | 520.14 | 524.79 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 528.65 | 525.94 | 525.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 538.05 | 529.39 | 527.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 526.00 | 532.25 | 530.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 526.00 | 532.25 | 530.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 526.00 | 532.25 | 530.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 526.00 | 532.25 | 530.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 531.60 | 532.12 | 530.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:15:00 | 534.75 | 532.12 | 530.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:00:00 | 533.00 | 532.30 | 530.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 15:15:00 | 526.25 | 533.25 | 533.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 15:15:00 | 526.25 | 533.25 | 533.33 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 538.65 | 534.33 | 533.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 12:15:00 | 551.15 | 538.82 | 536.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 09:15:00 | 549.35 | 553.85 | 548.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 549.35 | 553.85 | 548.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 549.35 | 553.85 | 548.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 550.10 | 553.85 | 548.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 547.85 | 552.65 | 548.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 548.95 | 552.65 | 548.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 547.25 | 551.57 | 548.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:30:00 | 546.85 | 551.57 | 548.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 545.15 | 550.29 | 547.85 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 539.70 | 545.90 | 546.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 535.70 | 541.75 | 544.01 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 567.80 | 546.36 | 545.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 577.20 | 555.53 | 550.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 570.80 | 571.82 | 563.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 570.80 | 571.82 | 563.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 570.80 | 571.82 | 563.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 14:00:00 | 574.00 | 570.59 | 565.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 573.45 | 571.56 | 566.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 573.80 | 573.18 | 568.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 575.00 | 573.18 | 568.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 573.45 | 574.43 | 571.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 558.70 | 571.13 | 571.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 558.70 | 571.13 | 571.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 539.95 | 559.29 | 564.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 541.45 | 540.78 | 549.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:45:00 | 541.60 | 540.78 | 549.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 547.20 | 543.31 | 548.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 543.45 | 545.85 | 549.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 552.30 | 545.81 | 547.48 | SL hit (close>static) qty=1.00 sl=551.40 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 561.25 | 550.67 | 549.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 570.15 | 557.85 | 553.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 574.85 | 576.09 | 569.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 13:15:00 | 571.90 | 574.71 | 570.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 571.90 | 574.71 | 570.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:45:00 | 569.85 | 574.71 | 570.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 574.95 | 574.76 | 570.91 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 10:15:00 | 568.05 | 569.73 | 569.95 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 573.80 | 570.18 | 570.10 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 14:15:00 | 568.20 | 569.68 | 569.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 15:15:00 | 567.20 | 569.18 | 569.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 571.00 | 569.55 | 569.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 571.00 | 569.55 | 569.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 571.00 | 569.55 | 569.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 569.45 | 569.55 | 569.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 580.60 | 571.76 | 570.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 582.70 | 573.94 | 571.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 573.75 | 574.29 | 572.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 573.75 | 574.29 | 572.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 571.10 | 573.40 | 572.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 571.10 | 573.40 | 572.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 573.40 | 573.40 | 572.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 569.75 | 573.40 | 572.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 574.45 | 576.16 | 574.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:45:00 | 573.05 | 576.16 | 574.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 571.65 | 575.26 | 574.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 571.65 | 575.26 | 574.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 571.60 | 574.53 | 574.40 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 571.45 | 573.91 | 574.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 561.15 | 571.36 | 572.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 567.85 | 567.73 | 570.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 15:00:00 | 567.85 | 567.73 | 570.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 566.65 | 566.82 | 569.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 567.70 | 566.82 | 569.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 567.70 | 566.87 | 568.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 567.40 | 566.87 | 568.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 571.75 | 567.85 | 568.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:30:00 | 572.15 | 567.85 | 568.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 571.00 | 568.48 | 569.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 571.00 | 568.48 | 569.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 577.10 | 570.45 | 569.96 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 563.70 | 568.53 | 569.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 560.65 | 566.96 | 568.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 11:15:00 | 565.05 | 562.92 | 565.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 565.05 | 562.92 | 565.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 565.05 | 562.92 | 565.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 565.80 | 562.92 | 565.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 561.60 | 562.66 | 565.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 564.35 | 562.66 | 565.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 538.85 | 539.46 | 547.80 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 560.00 | 550.36 | 550.14 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 543.75 | 549.76 | 550.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 542.50 | 547.78 | 548.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 548.95 | 547.96 | 548.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 548.95 | 547.96 | 548.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 548.95 | 547.96 | 548.84 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 555.25 | 549.66 | 549.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 562.70 | 553.05 | 551.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 545.55 | 554.61 | 553.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 545.55 | 554.61 | 553.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 545.55 | 554.61 | 553.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 545.55 | 554.61 | 553.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 543.00 | 552.29 | 552.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 539.95 | 552.29 | 552.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 542.30 | 550.29 | 551.26 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 561.00 | 552.48 | 551.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 568.10 | 561.22 | 556.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 567.20 | 569.20 | 564.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 567.20 | 569.20 | 564.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 598.45 | 605.30 | 600.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 598.45 | 605.30 | 600.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 597.10 | 603.66 | 599.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 603.35 | 603.66 | 599.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 595.65 | 601.92 | 600.38 | SL hit (close<static) qty=1.00 sl=596.15 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 607.00 | 602.88 | 602.57 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 601.50 | 605.41 | 605.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 595.65 | 601.48 | 603.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 593.25 | 590.93 | 594.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 601.00 | 593.07 | 594.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 600.65 | 593.07 | 594.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 605.30 | 595.51 | 595.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 605.30 | 595.51 | 595.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 601.75 | 596.76 | 596.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 15:15:00 | 606.40 | 600.17 | 598.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 592.55 | 604.43 | 605.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 14:15:00 | 590.55 | 598.19 | 601.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 605.90 | 599.36 | 601.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 610.15 | 601.52 | 602.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 613.60 | 601.52 | 602.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 612.00 | 603.61 | 603.25 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 591.50 | 601.05 | 602.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 581.65 | 597.17 | 600.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 590.00 | 586.95 | 591.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 590.00 | 586.95 | 591.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 592.10 | 587.98 | 591.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:30:00 | 593.40 | 587.98 | 591.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 592.35 | 588.85 | 591.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 592.85 | 588.85 | 591.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 591.60 | 589.40 | 591.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 590.35 | 589.40 | 591.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 590.00 | 589.80 | 591.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:30:00 | 586.50 | 583.99 | 587.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 599.70 | 587.60 | 588.22 | SL hit (close>static) qty=1.00 sl=596.70 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 596.55 | 589.39 | 588.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 612.90 | 596.08 | 592.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 596.05 | 600.34 | 596.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 599.10 | 600.09 | 596.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 601.05 | 599.87 | 596.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 602.50 | 598.86 | 597.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 586.30 | 596.61 | 596.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 586.30 | 596.61 | 596.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 580.65 | 587.75 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 595.35 | 588.36 | 590.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 589.65 | 588.62 | 590.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 587.45 | 588.63 | 590.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 577.15 | 576.45 | 576.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 577.15 | 576.45 | 576.44 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 575.00 | 576.16 | 576.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 15:15:00 | 574.00 | 575.73 | 576.10 | Break + close below crossover candle low |

### Cycle 87 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 581.95 | 576.97 | 576.63 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 574.25 | 577.08 | 577.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 13:15:00 | 572.60 | 575.82 | 576.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 569.95 | 569.78 | 572.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 569.95 | 569.78 | 572.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 574.20 | 570.46 | 571.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 574.20 | 570.46 | 571.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 575.05 | 571.38 | 572.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 579.25 | 571.38 | 572.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 578.65 | 573.37 | 572.89 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 569.05 | 572.36 | 572.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 563.70 | 567.77 | 569.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 569.30 | 566.71 | 568.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 567.20 | 566.81 | 568.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 577.40 | 566.81 | 568.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 570.05 | 567.46 | 568.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 567.50 | 567.67 | 568.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:00:00 | 567.55 | 567.64 | 568.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 565.85 | 568.11 | 568.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:00:00 | 566.85 | 567.50 | 568.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 566.90 | 567.38 | 567.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 567.00 | 567.38 | 567.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 564.50 | 564.00 | 565.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 568.35 | 564.00 | 565.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 566.85 | 564.57 | 565.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 566.40 | 564.57 | 565.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 567.90 | 565.23 | 565.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 567.80 | 565.23 | 565.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 564.95 | 566.94 | 567.04 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 570.00 | 567.56 | 567.31 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 563.00 | 566.90 | 567.11 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 567.75 | 565.53 | 565.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 573.10 | 567.04 | 566.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 575.20 | 575.30 | 571.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 575.05 | 577.71 | 575.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 575.05 | 577.71 | 575.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 578.10 | 577.79 | 575.67 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 572.70 | 574.80 | 574.86 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 576.05 | 575.05 | 574.97 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 566.45 | 573.34 | 574.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 561.15 | 569.70 | 572.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 557.05 | 554.23 | 558.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:45:00 | 556.45 | 554.23 | 558.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 559.70 | 555.33 | 558.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 559.70 | 555.33 | 558.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 558.45 | 555.95 | 558.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 556.55 | 557.28 | 558.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 562.95 | 558.42 | 559.23 | SL hit (close>static) qty=1.00 sl=561.65 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 11:15:00 | 582.30 | 559.50 | 553.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 10:15:00 | 575.00 | 579.86 | 572.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 11:00:00 | 575.00 | 579.86 | 572.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 566.05 | 577.10 | 572.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 566.05 | 577.10 | 572.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 568.20 | 575.32 | 571.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 569.30 | 572.80 | 571.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 565.20 | 569.45 | 569.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 565.20 | 569.45 | 569.84 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 572.70 | 570.16 | 570.10 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 567.75 | 569.69 | 569.95 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 575.40 | 570.83 | 570.45 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 566.50 | 570.80 | 571.18 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 572.55 | 571.18 | 571.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 575.00 | 572.79 | 572.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 577.90 | 572.60 | 572.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 579.40 | 572.44 | 572.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 577.35 | 585.08 | 583.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:45:00 | 578.10 | 583.74 | 582.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 580.50 | 582.56 | 582.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 580.50 | 582.56 | 582.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 587.60 | 582.49 | 582.20 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 579.75 | 581.94 | 581.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 578.35 | 581.22 | 581.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 578.05 | 577.78 | 579.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 578.05 | 577.78 | 579.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 557.00 | 555.20 | 560.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 543.75 | 550.01 | 555.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 546.45 | 544.03 | 548.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 545.55 | 547.18 | 548.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:00:00 | 546.05 | 546.96 | 548.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 555.00 | 548.25 | 548.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 554.25 | 548.25 | 548.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 551.30 | 548.86 | 548.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 545.00 | 549.60 | 549.81 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 551.60 | 549.58 | 549.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 558.20 | 551.67 | 550.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 555.65 | 556.00 | 553.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 555.65 | 556.00 | 553.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 553.00 | 555.97 | 554.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 556.35 | 555.97 | 554.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 550.10 | 554.93 | 554.67 | SL hit (close<static) qty=1.00 sl=550.50 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 550.55 | 554.05 | 554.30 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 561.80 | 555.60 | 554.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 13:15:00 | 570.10 | 564.43 | 560.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 571.50 | 571.85 | 568.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:45:00 | 572.30 | 571.85 | 568.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 566.80 | 570.94 | 568.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 566.80 | 570.94 | 568.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 576.55 | 572.06 | 569.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 584.00 | 573.59 | 571.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 578.55 | 584.44 | 584.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 578.55 | 584.44 | 584.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 577.35 | 583.02 | 583.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 578.50 | 577.82 | 580.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:30:00 | 578.25 | 577.82 | 580.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 572.15 | 571.37 | 573.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 563.90 | 571.37 | 573.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 576.55 | 570.49 | 571.65 | SL hit (close>static) qty=1.00 sl=573.80 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 569.85 | 566.62 | 566.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 581.10 | 569.87 | 568.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 570.85 | 573.15 | 570.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 570.90 | 572.70 | 570.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 570.90 | 572.70 | 570.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 571.95 | 572.55 | 570.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:15:00 | 573.00 | 572.55 | 570.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 558.20 | 569.23 | 569.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 558.20 | 569.23 | 569.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 556.05 | 566.60 | 568.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 550.55 | 550.40 | 556.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 550.55 | 550.40 | 556.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 548.95 | 549.73 | 554.86 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 556.75 | 554.48 | 554.48 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 551.55 | 554.16 | 554.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 550.05 | 552.81 | 553.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 539.05 | 536.57 | 540.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 541.00 | 537.45 | 540.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 541.00 | 537.45 | 540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 544.95 | 538.95 | 541.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 544.50 | 538.95 | 541.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 542.85 | 539.73 | 541.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:30:00 | 540.75 | 539.89 | 541.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 540.75 | 540.34 | 540.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 541.80 | 540.97 | 541.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 534.50 | 540.12 | 540.65 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 549.95 | 541.60 | 541.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 557.85 | 546.84 | 543.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 549.75 | 550.30 | 546.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:45:00 | 550.00 | 550.30 | 546.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 547.90 | 549.45 | 546.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 547.80 | 549.45 | 546.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 546.65 | 548.89 | 546.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 546.30 | 548.89 | 546.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 546.30 | 548.37 | 546.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 545.85 | 548.37 | 546.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 544.45 | 547.59 | 546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:15:00 | 547.40 | 547.59 | 546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 547.40 | 547.55 | 546.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 543.60 | 547.55 | 546.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 544.65 | 546.97 | 546.48 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 541.60 | 545.90 | 546.04 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 550.60 | 544.99 | 544.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 553.00 | 548.95 | 547.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 549.55 | 550.50 | 548.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 549.80 | 550.36 | 548.81 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 534.80 | 545.94 | 547.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 533.75 | 537.48 | 538.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 524.50 | 524.48 | 529.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:00:00 | 524.50 | 524.48 | 529.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 515.75 | 522.25 | 527.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 512.65 | 517.87 | 522.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 509.70 | 515.14 | 519.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 511.45 | 514.29 | 517.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 513.00 | 511.88 | 514.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 512.40 | 511.98 | 514.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 512.40 | 511.98 | 514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 515.05 | 512.60 | 514.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 515.05 | 512.60 | 514.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 514.05 | 512.89 | 514.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 513.00 | 513.42 | 514.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 512.25 | 513.19 | 514.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 511.75 | 512.93 | 514.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 512.55 | 512.61 | 513.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 512.50 | 512.59 | 513.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 512.55 | 512.59 | 513.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 511.65 | 511.66 | 512.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 507.90 | 511.46 | 512.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 508.00 | 511.06 | 512.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 508.00 | 508.22 | 509.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:15:00 | 507.75 | 508.22 | 509.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 506.80 | 507.94 | 509.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 510.00 | 511.95 | 512.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 506.45 | 510.09 | 511.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 510.35 | 509.12 | 510.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 510.60 | 509.42 | 510.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:15:00 | 510.80 | 509.42 | 510.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 514.80 | 510.49 | 510.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 514.80 | 510.49 | 510.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 514.05 | 511.21 | 511.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 10:15:00 | 515.50 | 512.29 | 511.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 13:15:00 | 530.10 | 530.19 | 526.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 527.45 | 529.64 | 526.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 527.45 | 529.64 | 526.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 527.45 | 529.64 | 526.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 528.25 | 529.37 | 526.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 525.60 | 528.97 | 526.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 529.20 | 529.02 | 527.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 532.65 | 529.74 | 527.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 533.35 | 529.80 | 527.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:00:00 | 531.65 | 530.17 | 528.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:45:00 | 530.40 | 530.16 | 528.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | SL hit (close<static) qty=1.00 sl=526.35 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 521.30 | 526.88 | 527.26 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 530.80 | 527.80 | 527.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 532.75 | 529.82 | 528.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 530.15 | 531.59 | 530.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 535.00 | 533.26 | 531.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 533.70 | 533.26 | 531.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 533.97 | 532.15 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 521.55 | 531.38 | 531.42 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 533.00 | 530.20 | 529.89 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 527.00 | 529.91 | 529.95 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 530.50 | 530.03 | 530.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 530.75 | 530.17 | 530.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 526.50 | 531.67 | 531.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 526.75 | 530.69 | 530.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 524.80 | 527.44 | 528.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 10:15:00 | 527.95 | 527.54 | 528.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 527.95 | 527.54 | 528.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 527.05 | 527.45 | 528.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 525.50 | 527.02 | 528.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 525.75 | 526.36 | 527.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 525.50 | 526.48 | 527.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 525.80 | 526.48 | 527.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 527.65 | 526.58 | 527.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 527.65 | 526.58 | 527.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | SL hit (close>static) qty=1.00 sl=529.10 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 15:15:00 | 532.35 | 528.48 | 528.04 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 525.15 | 527.40 | 527.67 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 535.90 | 529.10 | 528.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 15:15:00 | 538.10 | 530.90 | 529.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 530.15 | 530.72 | 529.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 529.20 | 530.42 | 529.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 529.20 | 530.42 | 529.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 533.10 | 530.95 | 529.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 532.40 | 530.95 | 529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 528.60 | 530.48 | 529.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 528.60 | 530.48 | 529.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 531.85 | 530.76 | 529.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 532.95 | 530.68 | 529.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 533.00 | 530.68 | 529.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 528.00 | 529.90 | 529.77 | SL hit (close<static) qty=1.00 sl=528.10 alert=retest2 |

### Cycle 138 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 529.25 | 531.22 | 531.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 525.20 | 530.02 | 530.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 521.00 | 520.98 | 523.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:45:00 | 521.05 | 520.98 | 523.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 522.20 | 521.29 | 522.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 522.20 | 521.29 | 522.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 522.10 | 520.90 | 521.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 522.10 | 520.90 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 520.05 | 520.73 | 521.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:15:00 | 523.05 | 520.73 | 521.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 523.05 | 521.19 | 521.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 521.30 | 521.19 | 521.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 520.15 | 520.98 | 521.74 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 526.00 | 521.46 | 521.21 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 521.05 | 521.88 | 521.93 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 523.70 | 521.95 | 521.93 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 521.10 | 521.83 | 521.89 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 522.50 | 521.85 | 521.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 526.00 | 522.68 | 522.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 521.70 | 522.48 | 522.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 521.50 | 522.29 | 522.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 521.50 | 522.29 | 522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 521.15 | 522.06 | 522.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 521.00 | 522.06 | 522.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 521.00 | 521.85 | 521.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 14:15:00 | 520.05 | 521.49 | 521.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 512.00 | 510.17 | 512.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 514.55 | 511.04 | 513.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 513.75 | 511.04 | 513.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 514.45 | 511.72 | 513.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 514.45 | 511.72 | 513.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 512.45 | 511.89 | 512.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 512.15 | 511.89 | 512.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 510.95 | 511.70 | 512.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 513.85 | 511.70 | 512.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 510.60 | 511.48 | 512.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 511.50 | 511.48 | 512.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 510.85 | 511.43 | 512.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 511.65 | 511.43 | 512.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 508.85 | 511.00 | 512.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 508.25 | 510.34 | 511.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 507.25 | 509.72 | 511.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 482.84 | 494.65 | 501.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 481.89 | 494.65 | 501.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 480.45 | 479.32 | 486.14 | SL hit (close>ema200) qty=0.50 sl=479.32 alert=retest2 |

### Cycle 145 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 478.00 | 471.16 | 471.11 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 470.00 | 470.93 | 471.01 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 471.60 | 471.06 | 471.06 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 469.45 | 470.74 | 470.91 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 487.65 | 473.77 | 472.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 490.55 | 477.13 | 473.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 480.10 | 482.44 | 478.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 483.75 | 483.65 | 480.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 482.00 | 483.65 | 480.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 481.10 | 483.63 | 480.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 481.10 | 483.63 | 480.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 475.05 | 481.92 | 480.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 488.05 | 483.70 | 481.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 483.75 | 482.81 | 481.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 481.95 | 482.81 | 481.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:45:00 | 482.05 | 482.68 | 481.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 486.65 | 483.47 | 481.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 491.40 | 484.88 | 482.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 478.65 | 481.40 | 483.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 481.95 | 481.22 | 483.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 481.95 | 481.22 | 483.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 485.00 | 481.97 | 483.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 484.60 | 481.97 | 483.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 488.95 | 483.37 | 484.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 488.95 | 483.37 | 484.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 490.55 | 484.80 | 484.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 493.40 | 488.04 | 486.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 491.40 | 491.53 | 489.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 491.40 | 491.53 | 489.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 493.90 | 492.01 | 489.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 492.00 | 492.01 | 489.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 491.40 | 491.89 | 489.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 489.20 | 491.89 | 489.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 487.65 | 491.04 | 489.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 487.65 | 491.04 | 489.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 486.30 | 490.09 | 489.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 485.30 | 490.09 | 489.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 479.70 | 487.70 | 488.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 477.00 | 485.56 | 487.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 473.85 | 472.30 | 475.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 11:00:00 | 473.85 | 472.30 | 475.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 479.05 | 473.65 | 476.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 479.05 | 473.65 | 476.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 481.45 | 475.21 | 476.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 482.35 | 475.21 | 476.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 483.10 | 477.35 | 477.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 483.10 | 477.35 | 477.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 483.65 | 478.61 | 477.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 493.00 | 481.85 | 479.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 493.15 | 493.78 | 488.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 497.90 | 493.78 | 488.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 494.75 | 497.01 | 493.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 494.75 | 497.01 | 493.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 492.50 | 496.10 | 493.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 491.05 | 496.10 | 493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 491.20 | 495.12 | 493.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 491.25 | 495.12 | 493.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 492.65 | 492.92 | 492.87 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 491.20 | 492.56 | 492.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 487.00 | 491.45 | 492.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 479.65 | 477.92 | 481.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 479.65 | 477.92 | 481.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 481.40 | 478.61 | 481.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 486.15 | 478.61 | 481.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 496.80 | 482.25 | 483.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 496.80 | 482.25 | 483.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 499.95 | 485.79 | 484.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 503.15 | 498.37 | 494.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 494.45 | 498.35 | 495.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 495.00 | 497.68 | 495.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 486.45 | 497.68 | 495.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 484.00 | 492.91 | 493.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 481.70 | 490.67 | 492.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 474.00 | 473.48 | 479.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 467.50 | 473.48 | 479.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 459.85 | 454.90 | 458.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 459.80 | 454.90 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 458.85 | 455.69 | 458.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 461.70 | 455.69 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 456.25 | 455.80 | 458.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 455.00 | 455.80 | 458.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:00:00 | 454.15 | 454.34 | 456.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 432.25 | 438.93 | 444.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 431.44 | 437.58 | 443.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 437.45 | 435.12 | 440.79 | SL hit (close>ema200) qty=0.50 sl=435.12 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 455.15 | 439.37 | 438.22 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 443.75 | 444.68 | 444.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 440.35 | 443.82 | 444.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 430.00 | 428.25 | 433.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 430.00 | 428.25 | 433.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 434.60 | 429.52 | 433.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 434.60 | 429.52 | 433.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 438.65 | 431.35 | 434.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 438.65 | 431.35 | 434.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 439.00 | 432.88 | 434.70 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 449.00 | 437.61 | 436.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 461.05 | 442.30 | 438.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 452.20 | 455.21 | 448.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 451.55 | 454.78 | 451.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 451.55 | 454.78 | 451.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 451.55 | 454.78 | 451.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 451.50 | 454.12 | 451.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 448.00 | 454.12 | 451.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 451.20 | 453.54 | 451.13 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 442.75 | 449.61 | 449.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 438.00 | 447.29 | 448.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 461.25 | 446.44 | 447.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 456.45 | 448.44 | 448.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 457.15 | 448.44 | 448.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 459.30 | 450.61 | 449.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 461.75 | 452.84 | 450.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 454.60 | 458.14 | 454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 461.55 | 458.82 | 455.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 461.90 | 458.82 | 455.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 463.70 | 459.34 | 455.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:45:00 | 462.40 | 461.60 | 458.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 462.50 | 461.78 | 459.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 468.50 | 463.12 | 460.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 469.15 | 463.82 | 460.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:30:00 | 469.75 | 465.17 | 461.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:00:00 | 469.45 | 466.86 | 462.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:00:00 | 471.40 | 467.77 | 463.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 464.95 | 467.96 | 465.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 464.85 | 467.96 | 465.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 465.70 | 467.51 | 465.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 472.30 | 467.51 | 465.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 483.55 | 486.76 | 487.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 488.75 | 485.55 | 486.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 487.00 | 485.84 | 486.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 14:45:00 | 486.45 | 486.08 | 486.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 482.70 | 486.26 | 486.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 486.65 | 486.34 | 486.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 492.80 | 488.22 | 487.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:45:00 | 489.80 | 495.50 | 492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 494.95 | 495.39 | 492.77 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 484.40 | 490.68 | 491.40 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 493.95 | 491.86 | 491.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 499.85 | 495.21 | 493.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 493.15 | 495.51 | 494.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 501.60 | 496.73 | 494.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 497.80 | 496.73 | 494.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 498.85 | 498.37 | 496.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 504.55 | 498.37 | 496.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 502.70 | 497.92 | 497.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 504.15 | 498.22 | 497.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 501.80 | 499.49 | 498.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 500.00 | 499.99 | 498.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 498.70 | 499.99 | 498.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 501.00 | 500.19 | 498.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 530.60 | 500.19 | 498.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 10:15:00 | 552.97 | 532.98 | 519.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 530.00 | 531.19 | 531.23 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-05-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-11 10:15:00 | 536.70 | 531.98 | 531.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-11 11:15:00 | 541.20 | 533.82 | 532.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 14:45:00 | 533.00 | 2024-05-29 09:15:00 | 506.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 14:45:00 | 533.00 | 2024-05-29 09:15:00 | 524.45 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2024-05-23 15:15:00 | 532.50 | 2024-05-29 09:15:00 | 505.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 15:15:00 | 532.50 | 2024-05-29 09:15:00 | 524.45 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2024-05-24 09:45:00 | 533.55 | 2024-05-29 09:15:00 | 506.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 09:45:00 | 533.55 | 2024-05-29 09:15:00 | 524.45 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2024-05-24 10:15:00 | 532.40 | 2024-05-29 09:15:00 | 505.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 10:15:00 | 532.40 | 2024-05-29 09:15:00 | 524.45 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2024-05-27 13:45:00 | 522.00 | 2024-05-29 13:15:00 | 529.05 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-05-28 10:00:00 | 519.50 | 2024-05-29 13:15:00 | 529.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-06-19 12:15:00 | 579.60 | 2024-06-25 09:15:00 | 583.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-07-01 14:00:00 | 552.50 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-07-02 10:45:00 | 554.50 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-07-02 11:30:00 | 554.10 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-03 09:45:00 | 553.60 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-03 12:15:00 | 549.85 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-03 13:30:00 | 550.00 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-05 10:15:00 | 549.90 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-05 10:45:00 | 549.90 | 2024-07-05 13:15:00 | 555.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-07-08 12:30:00 | 555.50 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 3.56% |
| BUY | retest2 | 2024-07-08 13:00:00 | 555.10 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2024-07-08 13:45:00 | 554.85 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 3.68% |
| BUY | retest2 | 2024-07-09 09:15:00 | 555.80 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 3.50% |
| BUY | retest2 | 2024-07-10 12:15:00 | 569.40 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2024-07-10 13:45:00 | 569.40 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2024-07-10 14:15:00 | 571.05 | 2024-07-16 13:15:00 | 575.25 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-07-23 09:15:00 | 568.20 | 2024-07-23 12:15:00 | 539.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:15:00 | 568.20 | 2024-07-23 14:15:00 | 565.50 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2024-08-07 10:15:00 | 558.25 | 2024-08-07 13:15:00 | 575.70 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-08-28 15:15:00 | 560.00 | 2024-08-29 09:15:00 | 579.25 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2024-09-04 14:45:00 | 593.00 | 2024-09-06 09:15:00 | 652.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 09:15:00 | 605.70 | 2024-09-06 10:15:00 | 666.27 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-18 09:15:00 | 673.25 | 2024-09-18 11:15:00 | 682.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-09-18 12:45:00 | 670.30 | 2024-09-23 09:15:00 | 680.85 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-09-25 11:15:00 | 682.50 | 2024-09-26 09:15:00 | 667.60 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-09-27 11:30:00 | 664.80 | 2024-09-27 14:15:00 | 677.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-09-27 12:45:00 | 663.50 | 2024-09-27 14:15:00 | 677.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-09-27 13:45:00 | 664.95 | 2024-09-27 14:15:00 | 677.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-10-17 09:15:00 | 701.00 | 2024-10-17 10:15:00 | 683.25 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-11-04 09:15:00 | 625.65 | 2024-11-04 12:15:00 | 641.50 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-11-04 12:45:00 | 632.25 | 2024-11-04 13:15:00 | 643.15 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-11-18 09:15:00 | 648.60 | 2024-11-25 09:15:00 | 660.35 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-11-18 12:30:00 | 657.65 | 2024-11-25 09:15:00 | 660.35 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-11-18 13:45:00 | 658.60 | 2024-11-25 09:15:00 | 660.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-11-19 10:30:00 | 657.05 | 2024-11-25 09:15:00 | 660.35 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-11-19 13:30:00 | 644.80 | 2024-11-25 09:15:00 | 660.35 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-11-29 14:00:00 | 682.20 | 2024-12-02 12:15:00 | 671.05 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-11-29 15:00:00 | 683.90 | 2024-12-02 12:15:00 | 671.05 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-12-02 10:00:00 | 683.35 | 2024-12-02 12:15:00 | 671.05 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-12-13 10:15:00 | 586.85 | 2024-12-18 09:15:00 | 596.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-13 12:00:00 | 589.85 | 2024-12-18 09:15:00 | 596.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-12-13 13:00:00 | 589.95 | 2024-12-18 09:15:00 | 596.25 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-13 13:45:00 | 589.65 | 2024-12-18 09:15:00 | 596.25 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-16 11:45:00 | 584.60 | 2024-12-18 10:15:00 | 602.80 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-12-16 13:15:00 | 585.30 | 2024-12-18 10:15:00 | 602.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-12-17 14:15:00 | 584.80 | 2024-12-18 10:15:00 | 602.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-12-17 14:45:00 | 585.00 | 2024-12-18 10:15:00 | 602.80 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-12-31 10:30:00 | 602.00 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-31 12:00:00 | 601.85 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-12-31 13:15:00 | 603.55 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-12-31 14:45:00 | 603.00 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-01-02 14:00:00 | 604.95 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-01-03 09:15:00 | 606.15 | 2025-01-03 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-01-17 13:15:00 | 647.10 | 2025-01-27 09:15:00 | 635.95 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-01-17 14:45:00 | 648.20 | 2025-01-27 09:15:00 | 635.95 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-01-29 12:30:00 | 609.55 | 2025-01-29 14:15:00 | 626.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-30 10:45:00 | 610.00 | 2025-02-03 11:15:00 | 610.05 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-01-30 11:30:00 | 609.15 | 2025-02-03 11:15:00 | 610.05 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-01-30 13:30:00 | 606.05 | 2025-02-03 11:15:00 | 610.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2025-02-10 09:15:00 | 634.30 | 2025-02-10 11:15:00 | 623.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-02-14 10:45:00 | 593.80 | 2025-02-18 09:15:00 | 564.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:45:00 | 593.80 | 2025-02-19 09:15:00 | 574.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-02-14 11:15:00 | 590.25 | 2025-02-20 09:15:00 | 560.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 11:00:00 | 587.90 | 2025-02-20 09:15:00 | 558.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:15:00 | 590.25 | 2025-02-21 11:15:00 | 559.60 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2025-02-17 11:00:00 | 587.90 | 2025-02-21 11:15:00 | 559.60 | STOP_HIT | 0.50 | 4.81% |
| BUY | retest2 | 2025-03-20 11:15:00 | 534.75 | 2025-03-21 15:15:00 | 526.25 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-20 12:00:00 | 533.00 | 2025-03-21 15:15:00 | 526.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-04-01 14:00:00 | 574.00 | 2025-04-04 09:15:00 | 558.70 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-04-02 10:15:00 | 573.45 | 2025-04-04 09:15:00 | 558.70 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-04-02 12:30:00 | 573.80 | 2025-04-04 09:15:00 | 558.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-04-02 13:00:00 | 575.00 | 2025-04-04 09:15:00 | 558.70 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-04-09 09:15:00 | 543.45 | 2025-04-09 13:15:00 | 552.30 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-05-20 09:15:00 | 603.35 | 2025-05-20 12:15:00 | 595.65 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-05-21 09:30:00 | 603.80 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-05-21 10:45:00 | 601.65 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-21 11:45:00 | 602.00 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 09:15:00 | 603.40 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-23 13:45:00 | 601.55 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-06-09 14:15:00 | 590.35 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-06-09 15:15:00 | 590.00 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-06-11 09:30:00 | 586.50 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-13 09:15:00 | 601.05 | 2025-06-16 09:15:00 | 586.30 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-06-13 14:30:00 | 602.50 | 2025-06-16 09:15:00 | 586.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-06-18 13:45:00 | 587.45 | 2025-06-24 13:15:00 | 577.15 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-07-07 11:15:00 | 567.50 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-07 12:00:00 | 567.55 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-07 15:15:00 | 565.85 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-07-08 10:00:00 | 566.85 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-24 09:45:00 | 556.55 | 2025-07-24 10:15:00 | 562.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-24 15:00:00 | 556.10 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-25 09:45:00 | 556.15 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-07-28 09:15:00 | 556.50 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-28 14:15:00 | 548.75 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-08-05 14:45:00 | 569.30 | 2025-08-06 10:15:00 | 565.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-14 09:15:00 | 577.90 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-08-18 09:15:00 | 579.40 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-08-20 10:15:00 | 577.35 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-08-20 10:45:00 | 578.10 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-29 13:45:00 | 543.75 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-09-01 15:00:00 | 546.45 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-02 13:45:00 | 545.55 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-02 15:00:00 | 546.05 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-09 09:15:00 | 556.35 | 2025-09-09 14:15:00 | 550.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-17 09:15:00 | 584.00 | 2025-09-22 13:15:00 | 578.55 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-26 09:15:00 | 563.90 | 2025-09-26 14:15:00 | 576.55 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-29 09:15:00 | 569.45 | 2025-09-29 14:15:00 | 574.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-29 10:45:00 | 570.95 | 2025-09-29 14:15:00 | 574.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-09-30 09:15:00 | 566.35 | 2025-10-01 15:15:00 | 567.60 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-01 14:30:00 | 563.15 | 2025-10-03 11:15:00 | 569.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-06 14:15:00 | 573.00 | 2025-10-07 09:15:00 | 558.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-10-16 10:30:00 | 540.75 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-17 11:00:00 | 540.75 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-17 14:00:00 | 541.80 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-11-11 09:30:00 | 512.65 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-11 14:30:00 | 509.70 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-12 10:45:00 | 511.45 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-13 10:15:00 | 513.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-14 09:15:00 | 513.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-14 10:00:00 | 512.25 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-14 10:30:00 | 511.75 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-14 13:00:00 | 512.55 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-11-17 14:30:00 | 507.90 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-18 09:15:00 | 508.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-18 13:45:00 | 508.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-18 14:15:00 | 507.75 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-02 12:00:00 | 532.65 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-12-02 13:15:00 | 533.35 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-12-02 14:00:00 | 531.65 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-02 14:45:00 | 530.40 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-16 14:00:00 | 525.50 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-17 10:15:00 | 525.75 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-17 11:30:00 | 525.50 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-17 12:15:00 | 525.80 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-12-19 14:30:00 | 532.95 | 2025-12-22 11:15:00 | 528.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-19 15:15:00 | 533.00 | 2025-12-22 11:15:00 | 528.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-22 15:00:00 | 535.75 | 2025-12-24 10:15:00 | 529.25 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-23 09:30:00 | 532.60 | 2025-12-24 10:15:00 | 529.25 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-14 13:30:00 | 508.25 | 2026-01-19 10:15:00 | 482.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 507.25 | 2026-01-19 10:15:00 | 481.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 508.25 | 2026-01-20 14:15:00 | 480.45 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2026-01-14 15:00:00 | 507.25 | 2026-01-20 14:15:00 | 480.45 | STOP_HIT | 0.50 | 5.28% |
| BUY | retest2 | 2026-02-02 09:30:00 | 488.05 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-02 11:30:00 | 483.75 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-02 12:15:00 | 481.95 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-02-02 12:45:00 | 482.05 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-02-02 14:30:00 | 491.40 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-03-10 12:15:00 | 455.00 | 2026-03-13 12:15:00 | 432.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:00:00 | 454.15 | 2026-03-13 13:15:00 | 431.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:15:00 | 455.00 | 2026-03-16 09:15:00 | 437.45 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2026-03-11 10:00:00 | 454.15 | 2026-03-16 09:15:00 | 437.45 | STOP_HIT | 0.50 | 3.68% |
| BUY | retest2 | 2026-04-02 12:15:00 | 461.90 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.38% |
| BUY | retest2 | 2026-04-02 13:15:00 | 463.70 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2026-04-06 10:45:00 | 462.40 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.27% |
| BUY | retest2 | 2026-04-06 12:00:00 | 462.50 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.24% |
| BUY | retest2 | 2026-04-06 13:30:00 | 469.15 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2026-04-06 14:30:00 | 469.75 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2026-04-07 10:00:00 | 469.45 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.69% |
| BUY | retest2 | 2026-04-07 11:00:00 | 471.40 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 472.30 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.06% |
| SELL | retest2 | 2026-04-21 14:45:00 | 486.45 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 482.70 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-22 09:45:00 | 486.65 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-04-29 09:15:00 | 504.55 | 2026-05-05 10:15:00 | 552.97 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2026-04-30 09:30:00 | 502.70 | 2026-05-05 10:15:00 | 551.98 | TARGET_HIT | 1.00 | 9.80% |
| BUY | retest2 | 2026-04-30 11:15:00 | 504.15 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-04-30 12:45:00 | 501.80 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2026-05-04 09:15:00 | 530.60 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | -0.11% |
