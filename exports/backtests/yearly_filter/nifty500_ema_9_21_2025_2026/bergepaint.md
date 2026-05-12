# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 515.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 29 |
| ALERT3 | 138 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 14 / 55
- **Target hits / Stop hits / Partials:** 3 / 62 / 4
- **Avg / median % per leg:** 0.24% / -0.79%
- **Sum % (uncompounded):** 16.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 3 | 11.1% | 3 | 24 | 0 | 0.30% | 8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 3 | 11.1% | 3 | 24 | 0 | 0.30% | 8.0% |
| SELL (all) | 42 | 11 | 26.2% | 0 | 38 | 4 | 0.21% | 8.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 11 | 26.2% | 0 | 38 | 4 | 0.21% | 8.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 14 | 20.3% | 3 | 62 | 4 | 0.24% | 16.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 542.30 | 540.62 | 540.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 546.65 | 543.04 | 542.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 571.10 | 572.07 | 565.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 571.10 | 572.07 | 565.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 567.10 | 571.07 | 565.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 567.10 | 571.07 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 567.70 | 570.40 | 566.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:15:00 | 564.95 | 570.40 | 566.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 564.45 | 569.21 | 565.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 564.45 | 569.21 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 563.50 | 568.07 | 565.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 563.00 | 568.07 | 565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 561.70 | 566.79 | 565.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 557.00 | 566.79 | 565.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 558.80 | 563.76 | 564.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 552.55 | 560.75 | 562.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 555.85 | 551.11 | 554.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 557.30 | 552.34 | 554.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 558.75 | 552.34 | 554.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 552.65 | 552.41 | 554.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 555.00 | 552.41 | 554.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 556.15 | 553.15 | 554.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 556.15 | 553.15 | 554.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 559.90 | 554.50 | 555.30 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 564.90 | 556.58 | 556.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 568.20 | 558.91 | 557.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 554.85 | 557.08 | 557.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 551.60 | 555.98 | 556.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 552.15 | 550.24 | 552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 552.00 | 549.90 | 551.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 552.00 | 549.90 | 551.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 550.05 | 549.93 | 551.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 547.15 | 549.93 | 551.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 549.50 | 549.95 | 551.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 553.85 | 550.73 | 551.49 | SL hit (close>static) qty=1.00 sl=552.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 559.80 | 553.31 | 552.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 561.60 | 557.70 | 555.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 554.05 | 558.02 | 556.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 553.95 | 557.21 | 555.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 553.95 | 557.21 | 555.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 552.70 | 556.30 | 555.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 553.60 | 556.30 | 555.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 550.95 | 554.55 | 554.90 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 569.60 | 557.56 | 556.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 579.70 | 568.21 | 564.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 576.75 | 577.19 | 572.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:15:00 | 575.95 | 577.19 | 572.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 571.10 | 577.04 | 574.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 573.25 | 577.04 | 574.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 571.70 | 575.97 | 574.12 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 571.50 | 572.90 | 573.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 567.55 | 571.69 | 572.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 576.00 | 572.51 | 572.35 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 571.05 | 572.15 | 572.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 568.85 | 571.27 | 571.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 569.25 | 566.94 | 568.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 565.10 | 566.57 | 568.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 558.90 | 566.57 | 568.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 564.05 | 565.63 | 567.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 13:30:00 | 563.95 | 565.89 | 567.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 570.10 | 566.73 | 567.52 | SL hit (close>static) qty=1.00 sl=570.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 572.65 | 568.30 | 568.12 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 09:15:00 | 562.00 | 567.96 | 568.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 560.30 | 563.04 | 564.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 544.95 | 541.05 | 547.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 541.65 | 541.05 | 547.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 547.55 | 542.35 | 547.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 547.55 | 542.35 | 547.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 548.60 | 543.60 | 547.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 548.60 | 543.60 | 547.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 550.90 | 545.06 | 547.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 550.90 | 545.06 | 547.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 553.45 | 549.02 | 548.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 563.50 | 554.91 | 552.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 15:15:00 | 590.00 | 590.41 | 585.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:15:00 | 586.40 | 590.41 | 585.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 585.30 | 589.39 | 585.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 585.00 | 589.39 | 585.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 589.65 | 589.44 | 585.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 585.00 | 589.44 | 585.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 595.80 | 598.27 | 595.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 595.05 | 598.27 | 595.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 595.85 | 597.79 | 595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 595.50 | 597.79 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 596.90 | 597.36 | 595.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 596.80 | 597.36 | 595.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 595.50 | 596.98 | 595.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 593.45 | 596.98 | 595.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 590.40 | 595.67 | 595.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 590.40 | 595.67 | 595.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 592.40 | 595.01 | 594.84 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 589.15 | 593.84 | 594.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 588.10 | 592.69 | 593.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 584.25 | 584.70 | 587.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 584.00 | 584.56 | 586.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 580.65 | 584.14 | 585.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 570.25 | 567.20 | 566.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 570.25 | 567.20 | 566.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 14:15:00 | 579.60 | 570.69 | 568.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 570.65 | 576.11 | 573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 573.35 | 575.56 | 573.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 570.30 | 575.56 | 573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 571.00 | 574.65 | 573.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 570.20 | 574.65 | 573.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 569.95 | 573.71 | 572.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 569.80 | 573.71 | 572.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 568.35 | 571.92 | 572.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 566.50 | 570.84 | 571.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 570.40 | 569.74 | 570.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 570.65 | 569.92 | 570.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 571.00 | 569.92 | 570.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 569.55 | 569.85 | 570.70 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 572.90 | 570.90 | 570.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 576.30 | 571.98 | 571.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 570.65 | 572.60 | 571.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 574.45 | 572.97 | 571.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 575.35 | 573.09 | 572.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 575.15 | 573.72 | 572.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 576.65 | 574.30 | 572.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 568.55 | 573.03 | 572.87 | SL hit (close<static) qty=1.00 sl=570.40 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 567.80 | 571.99 | 572.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 566.05 | 570.80 | 571.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 569.45 | 567.07 | 569.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 565.20 | 562.53 | 565.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 565.20 | 562.53 | 565.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 562.40 | 562.51 | 565.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 563.75 | 562.51 | 565.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 569.60 | 563.92 | 565.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 569.60 | 563.92 | 565.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 571.80 | 565.50 | 566.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 571.30 | 565.50 | 566.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 572.00 | 566.80 | 566.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 578.05 | 569.05 | 567.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 564.70 | 570.20 | 570.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 562.00 | 568.56 | 569.75 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 578.80 | 570.61 | 570.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 567.50 | 570.45 | 570.65 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 571.90 | 570.66 | 570.64 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 570.35 | 570.60 | 570.61 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 571.50 | 570.78 | 570.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 572.60 | 571.22 | 570.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 561.00 | 571.17 | 570.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 564.00 | 569.74 | 570.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 552.60 | 564.10 | 567.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 541.90 | 545.10 | 548.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 542.40 | 543.97 | 547.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 545.45 | 538.22 | 537.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 545.45 | 538.22 | 537.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 551.40 | 541.62 | 539.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 546.30 | 546.44 | 543.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 549.50 | 547.12 | 544.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 544.00 | 546.78 | 545.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 544.00 | 546.78 | 545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 543.15 | 546.05 | 544.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 543.15 | 546.05 | 544.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 538.30 | 543.65 | 543.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 536.25 | 542.17 | 543.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 530.20 | 529.34 | 532.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 530.20 | 529.34 | 532.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 531.70 | 529.77 | 532.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 532.40 | 529.77 | 532.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 532.70 | 530.36 | 532.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 532.70 | 530.36 | 532.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 535.45 | 531.38 | 532.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 535.45 | 531.38 | 532.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 534.55 | 532.01 | 532.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 532.30 | 532.01 | 532.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 528.90 | 531.39 | 532.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 527.95 | 531.39 | 532.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 526.10 | 530.00 | 531.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 527.45 | 529.64 | 531.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:15:00 | 526.90 | 529.64 | 531.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 534.75 | 528.63 | 529.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 534.75 | 528.63 | 529.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 531.30 | 529.17 | 529.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:00:00 | 527.50 | 529.70 | 530.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 534.00 | 530.58 | 530.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 534.00 | 530.58 | 530.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 15:15:00 | 535.70 | 531.77 | 530.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 541.90 | 541.95 | 538.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 11:30:00 | 541.45 | 541.95 | 538.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 545.70 | 542.71 | 539.94 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 534.50 | 540.51 | 541.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 532.70 | 538.95 | 540.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 542.45 | 539.41 | 540.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 538.70 | 539.27 | 540.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 541.55 | 539.27 | 540.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 540.90 | 539.60 | 540.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 534.80 | 539.60 | 540.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 536.85 | 538.96 | 539.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:00:00 | 536.60 | 538.73 | 539.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 534.00 | 539.27 | 539.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 535.85 | 533.58 | 534.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 535.55 | 533.58 | 534.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 535.45 | 533.96 | 534.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 536.85 | 533.96 | 534.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 538.35 | 534.83 | 535.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 538.35 | 534.83 | 535.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 538.90 | 535.65 | 535.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 538.90 | 535.65 | 535.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 540.20 | 536.56 | 535.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 540.80 | 545.78 | 542.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 539.20 | 544.46 | 542.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 543.35 | 544.46 | 542.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 543.65 | 542.55 | 542.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 536.80 | 543.49 | 543.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 536.80 | 543.49 | 543.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 535.65 | 540.31 | 541.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 537.00 | 536.40 | 538.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 537.00 | 536.40 | 538.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 535.10 | 536.08 | 538.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 533.80 | 535.82 | 537.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 533.00 | 534.74 | 536.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:15:00 | 532.10 | 535.01 | 535.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:30:00 | 533.15 | 533.66 | 533.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 534.40 | 533.81 | 533.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 534.25 | 533.81 | 533.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 536.40 | 534.32 | 534.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 536.40 | 534.32 | 534.19 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 530.60 | 533.97 | 534.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 530.10 | 532.52 | 533.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 516.75 | 516.51 | 521.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 516.85 | 516.51 | 521.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 518.55 | 516.96 | 520.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 519.10 | 516.96 | 520.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 518.70 | 517.31 | 520.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 516.60 | 517.31 | 520.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 529.00 | 518.91 | 518.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 529.00 | 518.91 | 518.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 539.15 | 528.06 | 523.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 533.50 | 533.74 | 528.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 533.50 | 533.74 | 528.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 532.45 | 532.52 | 530.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 535.90 | 532.62 | 530.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 12:15:00 | 529.50 | 532.00 | 531.55 | SL hit (close<static) qty=1.00 sl=529.55 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 525.50 | 530.18 | 530.77 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 533.05 | 530.88 | 530.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 533.80 | 531.73 | 531.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 536.45 | 534.56 | 533.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 532.00 | 533.25 | 533.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 532.00 | 533.25 | 533.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 527.70 | 532.14 | 532.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 533.00 | 532.33 | 532.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 536.70 | 533.20 | 532.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 539.80 | 541.02 | 538.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 539.80 | 541.02 | 538.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 539.80 | 540.78 | 538.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 540.90 | 540.52 | 538.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 540.10 | 540.44 | 539.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 537.25 | 539.80 | 538.88 | SL hit (close<static) qty=1.00 sl=537.50 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 540.55 | 543.33 | 543.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 539.35 | 542.30 | 542.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 539.60 | 539.20 | 540.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 550.80 | 540.93 | 540.64 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 532.25 | 540.08 | 540.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 530.85 | 535.49 | 538.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 536.75 | 533.94 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 535.00 | 534.15 | 536.36 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 549.85 | 537.85 | 537.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 566.50 | 553.16 | 547.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 575.20 | 577.84 | 572.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 575.20 | 577.84 | 572.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 584.90 | 587.68 | 584.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 583.65 | 587.68 | 584.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 581.30 | 586.40 | 583.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 581.30 | 586.40 | 583.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 580.50 | 585.22 | 583.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 580.50 | 585.22 | 583.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 585.35 | 585.25 | 583.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 587.90 | 585.80 | 584.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 578.40 | 584.78 | 585.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 578.40 | 584.78 | 585.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 576.50 | 579.97 | 582.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 12:15:00 | 571.50 | 570.26 | 574.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:00:00 | 571.50 | 570.26 | 574.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 571.20 | 570.45 | 574.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 569.70 | 572.45 | 574.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 568.10 | 571.90 | 573.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 541.22 | 548.38 | 552.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 539.70 | 548.38 | 552.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 540.00 | 538.31 | 543.81 | SL hit (close>ema200) qty=0.50 sl=538.31 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 547.65 | 544.89 | 544.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 549.10 | 545.73 | 545.16 | Break + close above crossover candle high |

### Cycle 46 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 538.00 | 544.76 | 544.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 532.95 | 537.13 | 538.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 535.80 | 534.85 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 536.00 | 535.08 | 536.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 535.75 | 535.08 | 536.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 538.30 | 535.72 | 536.61 | SL hit (close>static) qty=1.00 sl=537.65 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 537.95 | 537.09 | 537.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 542.05 | 538.44 | 537.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 543.30 | 543.48 | 541.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 544.90 | 544.70 | 542.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 549.80 | 546.02 | 543.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 540.95 | 547.00 | 547.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 540.95 | 547.00 | 547.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 542.59 | 544.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 533.20 | 530.67 | 535.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 533.20 | 530.67 | 535.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 532.15 | 531.08 | 534.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 534.35 | 531.08 | 534.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 536.25 | 532.11 | 534.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 536.25 | 532.11 | 534.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 535.00 | 532.69 | 534.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 533.85 | 532.69 | 534.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 507.16 | 514.78 | 517.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 513.20 | 512.11 | 514.51 | SL hit (close>ema200) qty=0.50 sl=512.11 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 517.95 | 515.96 | 515.71 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 514.90 | 518.28 | 518.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 13:15:00 | 509.45 | 513.91 | 515.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 465.00 | 464.50 | 470.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:30:00 | 465.00 | 464.50 | 470.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 455.45 | 461.48 | 466.18 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 471.30 | 466.99 | 466.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 478.70 | 470.89 | 468.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 472.50 | 477.23 | 474.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 473.95 | 476.57 | 474.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 473.95 | 476.57 | 474.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 474.75 | 476.21 | 474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 474.35 | 476.21 | 474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 473.10 | 475.59 | 474.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 474.00 | 475.59 | 474.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 473.55 | 475.18 | 474.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:15:00 | 472.50 | 475.18 | 474.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 472.95 | 474.73 | 474.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 472.45 | 474.73 | 474.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 473.85 | 474.56 | 474.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 462.10 | 474.56 | 474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 464.25 | 472.50 | 473.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 458.10 | 461.92 | 465.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 457.45 | 456.78 | 460.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 457.45 | 456.78 | 460.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 460.80 | 457.58 | 460.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 463.90 | 457.58 | 460.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 463.60 | 458.79 | 460.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 464.35 | 458.79 | 460.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 463.45 | 459.72 | 461.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 463.50 | 459.72 | 461.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 463.30 | 461.08 | 461.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 462.55 | 461.08 | 461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 463.90 | 461.64 | 461.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 460.40 | 461.52 | 461.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 460.05 | 461.52 | 461.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:45:00 | 459.60 | 459.58 | 460.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 459.75 | 459.76 | 460.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 464.25 | 460.66 | 460.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 464.25 | 460.66 | 460.61 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 459.40 | 461.52 | 461.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 457.65 | 459.61 | 460.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 461.70 | 459.17 | 459.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 464.30 | 461.29 | 460.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 462.55 | 461.94 | 460.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 460.70 | 461.73 | 460.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 460.70 | 461.73 | 460.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 461.60 | 461.70 | 460.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 462.30 | 461.82 | 461.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:45:00 | 463.65 | 462.45 | 461.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 462.85 | 462.53 | 461.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 457.30 | 460.97 | 461.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 457.30 | 460.97 | 461.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 14:15:00 | 455.40 | 459.26 | 460.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 461.35 | 459.48 | 460.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 461.10 | 459.80 | 460.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 459.15 | 459.35 | 459.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 436.19 | 454.90 | 457.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 451.00 | 450.07 | 453.16 | SL hit (close>ema200) qty=0.50 sl=450.07 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 437.00 | 435.96 | 435.96 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 433.70 | 435.80 | 435.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 429.90 | 433.92 | 434.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 405.00 | 400.96 | 409.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 405.00 | 400.96 | 409.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 416.85 | 404.61 | 409.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 414.40 | 404.61 | 409.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 416.55 | 407.00 | 410.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 415.30 | 407.00 | 410.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 417.35 | 411.78 | 411.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 417.35 | 411.78 | 411.74 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 410.00 | 413.21 | 413.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 409.05 | 412.05 | 412.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 415.70 | 413.24 | 413.23 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 405.85 | 413.76 | 413.86 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 415.20 | 411.37 | 411.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 423.20 | 414.51 | 412.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 413.70 | 420.48 | 417.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 414.50 | 419.28 | 417.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 413.25 | 419.28 | 417.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 416.50 | 417.79 | 417.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 416.50 | 417.79 | 417.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 414.30 | 417.09 | 416.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 414.30 | 417.09 | 416.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 414.20 | 416.51 | 416.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 408.25 | 414.86 | 415.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 409.90 | 409.71 | 412.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 409.90 | 409.71 | 412.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 414.00 | 410.57 | 412.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 417.80 | 410.57 | 412.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 417.40 | 411.93 | 413.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 413.60 | 412.93 | 413.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 416.80 | 413.70 | 413.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 416.80 | 413.70 | 413.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 417.70 | 415.09 | 414.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 10:30:00 | 415.65 | 415.93 | 414.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 15:15:00 | 457.22 | 449.91 | 443.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 471.95 | 475.82 | 476.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 470.70 | 474.80 | 475.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 463.70 | 463.15 | 466.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:30:00 | 464.40 | 463.15 | 466.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 463.15 | 463.55 | 465.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 465.40 | 463.55 | 465.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 466.75 | 464.19 | 465.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 462.95 | 464.19 | 465.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 460.50 | 463.45 | 465.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 458.85 | 460.17 | 462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 457.40 | 459.74 | 462.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 458.90 | 459.72 | 461.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 468.95 | 462.95 | 462.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 468.95 | 462.95 | 462.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 471.55 | 465.72 | 463.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 468.90 | 470.61 | 467.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 12:00:00 | 468.90 | 470.61 | 467.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 467.10 | 469.99 | 468.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 474.90 | 468.67 | 468.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 471.75 | 470.94 | 469.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 472.35 | 471.15 | 469.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 12:15:00 | 518.93 | 502.94 | 491.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-28 09:15:00 | 547.15 | 2025-05-28 10:15:00 | 553.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-28 10:15:00 | 549.50 | 2025-05-28 10:15:00 | 553.85 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-13 09:15:00 | 558.90 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 564.05 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-13 13:30:00 | 563.95 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-09 12:15:00 | 584.25 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-07-09 13:00:00 | 584.00 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-07-10 09:45:00 | 580.65 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2025-07-24 12:15:00 | 575.35 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-24 14:00:00 | 575.15 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-24 15:00:00 | 576.65 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-12 11:00:00 | 541.90 | 2025-08-18 12:15:00 | 545.45 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-12 12:30:00 | 542.40 | 2025-08-18 12:15:00 | 545.45 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-26 10:15:00 | 527.95 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-26 11:30:00 | 526.10 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-26 12:30:00 | 527.45 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-26 13:15:00 | 526.90 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-28 15:00:00 | 527.50 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-08 09:15:00 | 534.80 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-08 09:45:00 | 536.85 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-08 14:00:00 | 536.60 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-09 09:15:00 | 534.00 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-15 09:15:00 | 543.35 | 2025-09-16 14:15:00 | 536.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-15 14:00:00 | 543.65 | 2025-09-16 14:15:00 | 536.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-19 11:15:00 | 533.80 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-19 14:00:00 | 533.00 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-22 13:15:00 | 532.10 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-24 11:30:00 | 533.15 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-30 11:15:00 | 516.60 | 2025-10-01 14:15:00 | 529.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-10-07 14:15:00 | 535.90 | 2025-10-08 12:15:00 | 529.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-14 12:00:00 | 536.45 | 2025-10-15 12:15:00 | 532.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-21 13:45:00 | 540.90 | 2025-10-23 09:15:00 | 537.25 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-21 14:30:00 | 540.10 | 2025-10-23 09:15:00 | 537.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-23 10:45:00 | 541.80 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-23 12:15:00 | 540.60 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-10-23 13:45:00 | 543.00 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-10-24 12:30:00 | 542.35 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-24 14:00:00 | 542.95 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-10-27 09:30:00 | 542.50 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-29 11:45:00 | 544.75 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-29 14:45:00 | 545.25 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-30 12:30:00 | 544.90 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-30 15:00:00 | 545.00 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-20 12:30:00 | 587.90 | 2025-11-21 14:15:00 | 578.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-27 13:15:00 | 569.70 | 2025-12-08 11:15:00 | 541.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 15:15:00 | 568.10 | 2025-12-08 11:15:00 | 539.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 569.70 | 2025-12-09 12:15:00 | 540.00 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-27 15:15:00 | 568.10 | 2025-12-09 12:15:00 | 540.00 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-12-19 09:15:00 | 535.75 | 2025-12-19 09:15:00 | 538.30 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-24 12:00:00 | 549.80 | 2025-12-29 09:15:00 | 540.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-01 09:15:00 | 533.85 | 2026-01-12 09:15:00 | 507.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 533.85 | 2026-01-13 10:15:00 | 513.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2026-02-12 14:45:00 | 460.40 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-12 15:15:00 | 460.05 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-13 14:45:00 | 459.60 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-16 09:15:00 | 459.75 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-24 14:00:00 | 462.30 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-25 09:45:00 | 463.65 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-25 11:00:00 | 462.85 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-26 11:30:00 | 459.15 | 2026-03-02 09:15:00 | 436.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 459.15 | 2026-03-02 15:15:00 | 451.00 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2026-03-17 11:15:00 | 415.30 | 2026-03-17 13:15:00 | 417.35 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-04-01 11:45:00 | 413.60 | 2026-04-01 12:15:00 | 416.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-04-02 10:30:00 | 415.65 | 2026-04-10 15:15:00 | 457.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-29 10:00:00 | 458.85 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-29 10:30:00 | 457.40 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-30 09:30:00 | 458.90 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-05-05 15:15:00 | 474.90 | 2026-05-08 12:15:00 | 518.93 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2026-05-06 11:30:00 | 471.75 | 2026-05-08 12:15:00 | 519.59 | TARGET_HIT | 1.00 | 10.14% |
