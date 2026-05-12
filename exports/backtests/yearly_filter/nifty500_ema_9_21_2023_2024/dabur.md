# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 487.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 244 |
| ALERT1 | 156 |
| ALERT2 | 155 |
| ALERT2_SKIP | 90 |
| ALERT3 | 436 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 200 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 200 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 218 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 151
- **Target hits / Stop hits / Partials:** 6 / 200 / 12
- **Avg / median % per leg:** 0.30% / -0.49%
- **Sum % (uncompounded):** 64.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 24 | 32.0% | 2 | 73 | 0 | 0.05% | 3.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.36% | -1.4% |
| BUY @ 3rd Alert (retest2) | 74 | 24 | 32.4% | 2 | 72 | 0 | 0.07% | 5.1% |
| SELL (all) | 143 | 43 | 30.1% | 4 | 127 | 12 | 0.43% | 61.2% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.31% | -1.6% |
| SELL @ 3rd Alert (retest2) | 138 | 42 | 30.4% | 4 | 122 | 12 | 0.45% | 62.8% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.49% | -2.9% |
| retest2 (combined) | 212 | 66 | 31.1% | 6 | 194 | 12 | 0.32% | 67.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 523.55 | 526.69 | 526.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 520.55 | 523.36 | 524.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 10:15:00 | 522.15 | 520.70 | 522.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 10:15:00 | 522.15 | 520.70 | 522.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 522.15 | 520.70 | 522.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:45:00 | 522.30 | 520.70 | 522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 521.40 | 520.84 | 522.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:15:00 | 520.60 | 520.88 | 521.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 15:15:00 | 524.95 | 521.68 | 522.04 | SL hit (close>static) qty=1.00 sl=522.45 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 523.55 | 522.28 | 522.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 524.50 | 522.73 | 522.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 10:15:00 | 523.00 | 523.94 | 523.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 10:15:00 | 523.00 | 523.94 | 523.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 523.00 | 523.94 | 523.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 11:00:00 | 523.00 | 523.94 | 523.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 527.80 | 524.71 | 523.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 15:15:00 | 528.40 | 525.33 | 524.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 09:30:00 | 528.70 | 526.98 | 525.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 553.95 | 554.91 | 554.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 553.95 | 554.91 | 554.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 09:15:00 | 547.00 | 553.33 | 554.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 15:15:00 | 549.00 | 548.49 | 550.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-07 09:15:00 | 551.25 | 548.49 | 550.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 553.80 | 549.55 | 551.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 553.80 | 549.55 | 551.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 554.65 | 550.57 | 551.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 11:15:00 | 552.95 | 550.57 | 551.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 13:15:00 | 554.45 | 551.85 | 551.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 13:15:00 | 554.45 | 551.85 | 551.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 14:15:00 | 556.50 | 552.78 | 552.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 554.00 | 554.38 | 553.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 554.00 | 554.38 | 553.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 554.00 | 554.38 | 553.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 554.00 | 554.38 | 553.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 552.75 | 554.05 | 553.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 552.75 | 554.05 | 553.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 552.00 | 553.64 | 553.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:45:00 | 552.55 | 553.64 | 553.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 551.90 | 553.29 | 553.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 555.90 | 553.03 | 552.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 09:15:00 | 550.10 | 552.45 | 552.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 09:15:00 | 550.10 | 552.45 | 552.68 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 553.90 | 552.53 | 552.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 14:15:00 | 554.10 | 552.84 | 552.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 10:15:00 | 567.00 | 567.24 | 564.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 11:00:00 | 567.00 | 567.24 | 564.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 569.45 | 570.13 | 568.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 14:15:00 | 569.95 | 570.13 | 568.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 15:00:00 | 569.90 | 570.08 | 568.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 567.10 | 569.15 | 568.15 | SL hit (close<static) qty=1.00 sl=567.65 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 566.75 | 568.69 | 568.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 563.35 | 567.62 | 568.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 563.00 | 561.96 | 564.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 13:00:00 | 563.00 | 561.96 | 564.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 562.50 | 562.06 | 564.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 13:45:00 | 563.15 | 562.06 | 564.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 563.00 | 562.25 | 563.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:45:00 | 565.50 | 562.25 | 563.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 15:15:00 | 564.00 | 562.60 | 563.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:15:00 | 563.70 | 562.60 | 563.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 562.45 | 562.57 | 563.83 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 568.50 | 565.06 | 564.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 569.95 | 566.51 | 565.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 09:15:00 | 566.35 | 566.85 | 565.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 566.35 | 566.85 | 565.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 566.35 | 566.85 | 565.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 565.50 | 566.85 | 565.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 564.75 | 566.43 | 565.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 564.90 | 566.43 | 565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 567.25 | 566.60 | 565.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 12:15:00 | 568.90 | 566.60 | 565.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 578.55 | 585.50 | 585.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 578.55 | 585.50 | 585.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 13:15:00 | 577.10 | 582.70 | 584.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 578.80 | 574.02 | 576.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 578.80 | 574.02 | 576.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 578.80 | 574.02 | 576.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 578.95 | 574.02 | 576.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 579.10 | 575.04 | 577.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 579.10 | 575.04 | 577.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 581.80 | 576.39 | 577.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:00:00 | 581.80 | 576.39 | 577.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 581.70 | 578.46 | 578.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 582.80 | 579.84 | 578.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 15:15:00 | 581.20 | 581.24 | 580.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:15:00 | 585.05 | 581.24 | 580.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 582.90 | 581.57 | 580.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 577.10 | 581.27 | 580.70 | SL hit (close<ema400) qty=1.00 sl=580.70 alert=retest1 |

### Cycle 11 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 577.20 | 579.87 | 580.12 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 583.35 | 580.80 | 580.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 586.30 | 582.63 | 581.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 10:15:00 | 585.70 | 586.11 | 583.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 11:00:00 | 585.70 | 586.11 | 583.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 582.85 | 585.28 | 583.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:45:00 | 582.35 | 585.28 | 583.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 581.60 | 584.55 | 583.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 582.40 | 584.55 | 583.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 581.60 | 583.53 | 583.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:15:00 | 579.50 | 583.53 | 583.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 09:15:00 | 575.85 | 581.99 | 582.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 10:15:00 | 574.25 | 580.44 | 581.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 579.25 | 577.71 | 579.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 09:15:00 | 579.25 | 577.71 | 579.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 579.25 | 577.71 | 579.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 09:30:00 | 579.90 | 577.71 | 579.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 575.90 | 577.35 | 579.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 14:45:00 | 574.25 | 576.20 | 578.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 09:15:00 | 573.90 | 576.02 | 577.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:15:00 | 573.65 | 575.88 | 577.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 15:15:00 | 579.75 | 577.00 | 577.31 | SL hit (close>static) qty=1.00 sl=579.55 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 574.05 | 569.13 | 568.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 14:15:00 | 574.75 | 572.07 | 570.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 14:15:00 | 574.20 | 574.32 | 572.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 14:15:00 | 574.20 | 574.32 | 572.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 574.20 | 574.32 | 572.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:30:00 | 572.45 | 574.32 | 572.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 572.25 | 573.91 | 572.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 574.05 | 573.91 | 572.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 576.25 | 574.38 | 573.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 12:45:00 | 578.25 | 575.50 | 573.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 10:15:00 | 573.90 | 575.26 | 575.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 10:15:00 | 573.90 | 575.26 | 575.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 12:15:00 | 569.30 | 573.67 | 574.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 568.60 | 560.30 | 563.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 568.60 | 560.30 | 563.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 568.60 | 560.30 | 563.45 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 569.75 | 565.40 | 565.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 572.05 | 566.73 | 565.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 562.70 | 566.55 | 566.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 562.70 | 566.55 | 566.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 562.70 | 566.55 | 566.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:45:00 | 561.35 | 566.55 | 566.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 563.00 | 565.84 | 565.81 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 11:15:00 | 563.35 | 565.34 | 565.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 11:15:00 | 559.90 | 563.48 | 564.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 10:15:00 | 568.60 | 563.74 | 564.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 10:15:00 | 568.60 | 563.74 | 564.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 568.60 | 563.74 | 564.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:00:00 | 568.60 | 563.74 | 564.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 11:15:00 | 568.25 | 564.64 | 564.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 13:15:00 | 570.60 | 566.67 | 565.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 570.80 | 570.91 | 568.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 13:45:00 | 569.65 | 570.91 | 568.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 563.30 | 569.46 | 568.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 563.30 | 569.46 | 568.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 562.75 | 568.12 | 568.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:45:00 | 563.10 | 568.12 | 568.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 564.15 | 567.32 | 567.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 12:15:00 | 560.95 | 563.78 | 565.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 565.25 | 563.88 | 565.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 14:15:00 | 565.25 | 563.88 | 565.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 565.25 | 563.88 | 565.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 565.25 | 563.88 | 565.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 564.60 | 564.02 | 565.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 561.60 | 564.02 | 565.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 10:45:00 | 561.95 | 563.15 | 564.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 09:30:00 | 562.60 | 563.62 | 564.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 10:15:00 | 562.50 | 563.62 | 564.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 563.05 | 562.95 | 563.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:45:00 | 563.75 | 562.95 | 563.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 561.35 | 562.63 | 563.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-18 14:15:00 | 565.70 | 563.73 | 563.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 565.70 | 563.73 | 563.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 568.10 | 565.77 | 564.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 570.25 | 572.07 | 569.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 10:00:00 | 570.25 | 572.07 | 569.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 568.80 | 571.41 | 569.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:45:00 | 568.35 | 571.41 | 569.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 567.15 | 570.56 | 569.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:00:00 | 567.15 | 570.56 | 569.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 565.85 | 569.62 | 568.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 565.85 | 569.62 | 568.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 567.05 | 569.09 | 568.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 567.05 | 569.09 | 568.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 566.10 | 568.49 | 568.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 568.75 | 568.49 | 568.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 09:15:00 | 567.70 | 568.33 | 568.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 09:15:00 | 567.70 | 568.33 | 568.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 13:15:00 | 566.55 | 567.63 | 568.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 15:15:00 | 551.00 | 550.59 | 554.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-30 09:15:00 | 553.95 | 550.59 | 554.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 554.95 | 551.46 | 554.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:00:00 | 554.95 | 551.46 | 554.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 555.15 | 552.20 | 554.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 11:30:00 | 554.35 | 552.40 | 554.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 15:15:00 | 554.25 | 553.37 | 554.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 10:00:00 | 553.05 | 553.45 | 554.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 13:15:00 | 556.85 | 553.32 | 553.69 | SL hit (close>static) qty=1.00 sl=555.85 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 554.40 | 553.53 | 553.50 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 09:15:00 | 552.00 | 553.22 | 553.37 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 12:15:00 | 555.60 | 553.55 | 553.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 13:15:00 | 555.85 | 554.01 | 553.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 11:15:00 | 560.00 | 560.01 | 558.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 12:00:00 | 560.00 | 560.01 | 558.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 564.80 | 567.85 | 566.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 564.95 | 567.85 | 566.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 565.05 | 567.29 | 566.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 564.00 | 567.29 | 566.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 563.25 | 565.99 | 566.05 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 567.30 | 565.60 | 565.47 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 10:15:00 | 562.55 | 565.15 | 565.39 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 15:15:00 | 567.40 | 565.51 | 565.43 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 564.40 | 565.23 | 565.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 555.20 | 563.14 | 564.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 14:15:00 | 560.60 | 558.90 | 560.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 14:15:00 | 560.60 | 558.90 | 560.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 560.60 | 558.90 | 560.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 560.60 | 558.90 | 560.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 559.75 | 559.07 | 560.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 559.95 | 559.07 | 560.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 559.35 | 559.12 | 560.37 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 561.50 | 560.80 | 560.74 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 09:15:00 | 555.45 | 559.73 | 560.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 14:15:00 | 554.25 | 557.84 | 559.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 15:15:00 | 555.95 | 555.77 | 557.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 15:15:00 | 555.95 | 555.77 | 557.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 555.95 | 555.77 | 557.08 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 561.00 | 558.09 | 557.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 15:15:00 | 561.70 | 558.81 | 558.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 09:15:00 | 557.75 | 558.60 | 558.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 557.75 | 558.60 | 558.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 557.75 | 558.60 | 558.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 556.90 | 558.60 | 558.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 555.85 | 558.05 | 557.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 555.85 | 558.05 | 557.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 554.30 | 557.30 | 557.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 551.05 | 556.05 | 556.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 13:15:00 | 551.40 | 551.31 | 553.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 13:30:00 | 551.00 | 551.31 | 553.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 549.80 | 550.99 | 552.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 553.05 | 550.99 | 552.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 549.15 | 550.62 | 552.46 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 14:15:00 | 555.20 | 553.16 | 552.95 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 10:15:00 | 547.90 | 552.03 | 552.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 14:15:00 | 547.00 | 549.31 | 550.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 544.15 | 543.55 | 545.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 10:30:00 | 544.90 | 543.55 | 545.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 544.90 | 543.98 | 545.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:45:00 | 543.70 | 544.18 | 545.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 14:30:00 | 543.90 | 544.40 | 544.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 14:15:00 | 541.35 | 539.67 | 539.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 541.35 | 539.67 | 539.52 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 537.45 | 539.13 | 539.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 537.00 | 538.71 | 539.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 14:15:00 | 525.35 | 522.80 | 526.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-20 15:00:00 | 525.35 | 522.80 | 526.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 524.30 | 523.48 | 525.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 14:45:00 | 522.30 | 523.17 | 525.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 10:00:00 | 522.80 | 522.45 | 524.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:15:00 | 522.00 | 522.58 | 524.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 13:15:00 | 524.55 | 519.82 | 519.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 524.55 | 519.82 | 519.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 527.85 | 523.95 | 522.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 10:15:00 | 525.55 | 526.88 | 525.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 10:15:00 | 525.55 | 526.88 | 525.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 525.55 | 526.88 | 525.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 525.55 | 526.88 | 525.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 523.65 | 526.23 | 524.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 523.65 | 526.23 | 524.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 519.50 | 524.89 | 524.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 519.50 | 524.89 | 524.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 518.50 | 523.61 | 523.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 517.30 | 522.35 | 523.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 11:15:00 | 521.90 | 520.84 | 522.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 521.90 | 520.84 | 522.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 521.90 | 520.84 | 522.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:45:00 | 521.75 | 520.84 | 522.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 524.00 | 521.48 | 522.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 524.00 | 521.48 | 522.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 524.55 | 522.09 | 522.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 524.55 | 522.09 | 522.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 529.90 | 523.65 | 523.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 532.90 | 525.50 | 524.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 537.05 | 537.15 | 534.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 09:45:00 | 537.05 | 537.15 | 534.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 536.60 | 536.55 | 534.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 13:45:00 | 533.75 | 536.55 | 534.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 536.15 | 536.47 | 534.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:45:00 | 535.50 | 536.47 | 534.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 535.50 | 536.28 | 534.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:15:00 | 537.80 | 536.28 | 534.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 537.55 | 536.53 | 535.15 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 13:15:00 | 533.85 | 535.76 | 535.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 529.60 | 534.02 | 535.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 533.50 | 533.23 | 534.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 15:15:00 | 533.50 | 533.23 | 534.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 533.50 | 533.23 | 534.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 535.70 | 533.23 | 534.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 535.80 | 533.75 | 534.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 531.60 | 533.75 | 534.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 12:15:00 | 533.10 | 530.23 | 530.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 12:15:00 | 533.10 | 530.23 | 530.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 13:15:00 | 534.95 | 531.17 | 530.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 09:15:00 | 540.35 | 541.77 | 539.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 540.35 | 541.77 | 539.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 540.35 | 541.77 | 539.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 539.70 | 541.77 | 539.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 542.00 | 542.22 | 540.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 14:45:00 | 540.10 | 542.22 | 540.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 540.00 | 541.77 | 540.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 539.25 | 541.77 | 540.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 540.45 | 541.51 | 540.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 15:15:00 | 541.95 | 540.39 | 540.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 536.60 | 539.88 | 540.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 09:15:00 | 536.60 | 539.88 | 540.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 534.15 | 537.01 | 538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 13:15:00 | 536.55 | 535.58 | 536.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 13:15:00 | 536.55 | 535.58 | 536.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 536.55 | 535.58 | 536.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:00:00 | 536.55 | 535.58 | 536.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 534.05 | 535.28 | 536.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 09:45:00 | 532.25 | 534.64 | 536.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 11:15:00 | 533.30 | 534.40 | 535.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:00:00 | 533.15 | 534.15 | 535.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 11:15:00 | 537.25 | 535.45 | 535.56 | SL hit (close>static) qty=1.00 sl=537.10 alert=retest2 |

### Cycle 44 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 538.35 | 536.17 | 535.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 540.25 | 536.98 | 536.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 12:15:00 | 550.00 | 550.07 | 547.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 12:45:00 | 550.00 | 550.07 | 547.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 545.15 | 549.29 | 548.08 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 12:15:00 | 545.20 | 547.46 | 547.47 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 11:15:00 | 548.00 | 547.36 | 547.31 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 546.70 | 547.23 | 547.26 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 13:15:00 | 548.00 | 547.38 | 547.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 549.25 | 547.94 | 547.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 10:15:00 | 545.90 | 547.53 | 547.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 10:15:00 | 545.90 | 547.53 | 547.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 545.90 | 547.53 | 547.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:45:00 | 545.15 | 547.53 | 547.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 547.45 | 547.52 | 547.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 12:15:00 | 547.90 | 547.52 | 547.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 12:15:00 | 548.30 | 548.82 | 548.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 13:00:00 | 548.00 | 548.65 | 548.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 09:15:00 | 543.90 | 547.95 | 548.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 543.90 | 547.95 | 548.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 12:15:00 | 542.75 | 545.24 | 546.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 15:15:00 | 542.85 | 541.19 | 542.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 15:15:00 | 542.85 | 541.19 | 542.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 542.85 | 541.19 | 542.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:15:00 | 540.70 | 541.19 | 542.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 540.00 | 540.95 | 542.45 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 13:15:00 | 548.90 | 544.29 | 543.68 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 535.30 | 542.80 | 543.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 531.25 | 538.25 | 541.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 09:15:00 | 532.30 | 530.62 | 533.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 09:15:00 | 532.30 | 530.62 | 533.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 532.30 | 530.62 | 533.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:45:00 | 532.65 | 530.62 | 533.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 11:15:00 | 532.15 | 531.17 | 532.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 12:15:00 | 531.95 | 531.17 | 532.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 10:30:00 | 531.45 | 531.15 | 532.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 14:15:00 | 534.35 | 532.42 | 532.48 | SL hit (close>static) qty=1.00 sl=533.05 alert=retest2 |

### Cycle 52 — BUY (started 2023-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 15:15:00 | 535.05 | 532.95 | 532.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 11:15:00 | 538.35 | 534.76 | 533.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 557.00 | 557.52 | 551.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 15:00:00 | 557.00 | 557.52 | 551.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 552.65 | 555.79 | 552.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:30:00 | 552.80 | 555.79 | 552.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 553.55 | 555.34 | 552.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:30:00 | 555.80 | 555.51 | 553.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 12:15:00 | 549.95 | 556.70 | 554.95 | SL hit (close<static) qty=1.00 sl=552.35 alert=retest2 |

### Cycle 53 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 551.00 | 553.67 | 553.82 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 558.30 | 554.59 | 554.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 10:15:00 | 561.70 | 556.01 | 554.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 560.15 | 566.18 | 563.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 560.15 | 566.18 | 563.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 560.15 | 566.18 | 563.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:45:00 | 560.05 | 566.18 | 563.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 555.30 | 564.00 | 562.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 555.30 | 564.00 | 562.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 557.75 | 561.74 | 561.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 13:15:00 | 556.40 | 560.67 | 561.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 15:15:00 | 549.50 | 548.08 | 551.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-11 09:15:00 | 550.95 | 548.08 | 551.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 551.00 | 548.67 | 551.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:00:00 | 551.00 | 548.67 | 551.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 553.50 | 549.63 | 551.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:45:00 | 553.80 | 549.63 | 551.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 553.55 | 550.42 | 551.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:30:00 | 553.40 | 550.42 | 551.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 552.00 | 550.34 | 551.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 15:00:00 | 552.00 | 550.34 | 551.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 550.80 | 550.43 | 551.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 551.70 | 550.43 | 551.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 549.85 | 550.32 | 551.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:30:00 | 551.75 | 550.32 | 551.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 11:15:00 | 548.00 | 549.80 | 550.72 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 554.20 | 550.94 | 550.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 10:15:00 | 560.35 | 555.19 | 553.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 552.60 | 556.61 | 555.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 552.60 | 556.61 | 555.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 552.60 | 556.61 | 555.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:30:00 | 555.55 | 556.61 | 555.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 551.25 | 555.54 | 554.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:30:00 | 548.85 | 555.54 | 554.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 548.05 | 554.04 | 554.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 539.75 | 549.51 | 551.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 14:15:00 | 542.65 | 541.83 | 544.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 14:45:00 | 542.05 | 541.83 | 544.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 537.95 | 541.24 | 543.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 09:45:00 | 535.50 | 538.37 | 540.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 09:15:00 | 535.10 | 533.80 | 534.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 13:15:00 | 535.25 | 533.47 | 533.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 535.25 | 533.47 | 533.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 537.00 | 534.44 | 533.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 534.30 | 534.41 | 533.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 10:15:00 | 534.30 | 534.41 | 533.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 534.30 | 534.41 | 533.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 10:45:00 | 534.85 | 534.41 | 533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 535.35 | 534.60 | 534.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 11:45:00 | 534.80 | 534.60 | 534.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 536.20 | 535.61 | 534.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 536.20 | 535.61 | 534.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 536.50 | 535.79 | 534.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:15:00 | 538.80 | 535.92 | 535.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 540.35 | 536.76 | 535.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 09:15:00 | 540.90 | 545.53 | 545.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 540.90 | 545.53 | 545.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 10:15:00 | 539.15 | 544.25 | 545.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 12:15:00 | 536.75 | 534.26 | 536.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 12:15:00 | 536.75 | 534.26 | 536.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 536.75 | 534.26 | 536.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 13:00:00 | 536.75 | 534.26 | 536.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 536.50 | 534.70 | 536.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 11:00:00 | 534.90 | 535.92 | 536.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 14:15:00 | 539.30 | 536.23 | 535.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 14:15:00 | 539.30 | 536.23 | 535.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 09:15:00 | 540.50 | 537.32 | 536.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 12:15:00 | 538.35 | 538.38 | 537.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 12:15:00 | 538.35 | 538.38 | 537.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 538.35 | 538.38 | 537.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 538.35 | 538.38 | 537.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 538.75 | 538.45 | 537.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:45:00 | 538.45 | 538.45 | 537.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 536.80 | 538.12 | 537.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 536.80 | 538.12 | 537.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 536.35 | 537.77 | 537.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 533.30 | 537.77 | 537.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 535.55 | 537.21 | 537.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 12:00:00 | 535.55 | 537.21 | 537.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 12:15:00 | 536.20 | 537.01 | 537.03 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 13:15:00 | 537.65 | 537.14 | 537.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 14:15:00 | 539.55 | 537.62 | 537.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 15:15:00 | 536.50 | 537.40 | 537.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 15:15:00 | 536.50 | 537.40 | 537.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 536.50 | 537.40 | 537.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:15:00 | 536.70 | 537.40 | 537.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 537.75 | 537.47 | 537.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 10:15:00 | 537.90 | 537.47 | 537.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 11:30:00 | 538.50 | 538.13 | 537.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 10:15:00 | 541.15 | 544.75 | 545.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 541.15 | 544.75 | 545.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 10:15:00 | 539.55 | 541.91 | 543.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 536.00 | 532.94 | 536.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 536.00 | 532.94 | 536.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 536.00 | 532.94 | 536.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 536.00 | 532.94 | 536.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 535.35 | 533.42 | 536.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:30:00 | 536.90 | 533.42 | 536.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 535.70 | 534.24 | 536.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:30:00 | 537.00 | 534.24 | 536.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 534.70 | 534.33 | 535.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 14:15:00 | 534.50 | 534.33 | 535.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 09:15:00 | 538.25 | 535.52 | 536.08 | SL hit (close>static) qty=1.00 sl=536.60 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 13:15:00 | 539.20 | 536.41 | 536.32 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 533.25 | 535.84 | 536.09 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 538.80 | 536.01 | 536.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 543.45 | 538.02 | 536.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 15:15:00 | 539.35 | 540.21 | 538.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 15:15:00 | 539.35 | 540.21 | 538.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 539.35 | 540.21 | 538.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 538.70 | 540.21 | 538.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 540.00 | 540.17 | 538.89 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 536.55 | 538.36 | 538.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 12:15:00 | 532.85 | 537.26 | 537.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 14:15:00 | 537.30 | 537.11 | 537.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 537.30 | 537.11 | 537.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 537.30 | 537.11 | 537.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 15:00:00 | 537.30 | 537.11 | 537.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 533.25 | 532.44 | 533.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 533.25 | 532.44 | 533.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 536.95 | 533.34 | 534.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 536.95 | 533.34 | 534.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 535.25 | 533.72 | 534.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:30:00 | 535.85 | 533.72 | 534.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 536.05 | 534.60 | 534.58 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 534.00 | 534.48 | 534.53 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 536.45 | 534.88 | 534.70 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 532.15 | 534.54 | 534.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 529.40 | 533.51 | 534.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 525.55 | 523.70 | 526.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 525.55 | 523.70 | 526.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 527.30 | 524.79 | 526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:30:00 | 527.95 | 524.79 | 526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 527.80 | 525.39 | 526.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 527.60 | 525.39 | 526.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 525.85 | 525.49 | 526.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:45:00 | 527.65 | 525.49 | 526.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 529.90 | 526.37 | 527.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 529.90 | 526.37 | 527.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 527.55 | 526.60 | 527.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:00:00 | 527.45 | 526.77 | 527.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 12:15:00 | 527.15 | 527.10 | 527.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 12:15:00 | 528.30 | 527.34 | 527.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 528.30 | 527.34 | 527.34 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 13:15:00 | 527.05 | 527.28 | 527.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 522.35 | 526.20 | 526.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 13:15:00 | 524.80 | 524.77 | 525.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-18 14:00:00 | 524.80 | 524.77 | 525.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 518.30 | 523.15 | 524.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 10:00:00 | 516.95 | 519.57 | 521.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 10:30:00 | 515.40 | 519.13 | 521.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 12:15:00 | 524.85 | 521.81 | 521.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 524.85 | 521.81 | 521.53 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 13:15:00 | 519.35 | 521.74 | 521.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 10:15:00 | 516.25 | 520.04 | 521.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 12:15:00 | 519.80 | 519.55 | 520.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 12:45:00 | 519.75 | 519.55 | 520.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 521.35 | 519.91 | 520.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:00:00 | 521.35 | 519.91 | 520.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 521.05 | 520.14 | 520.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:30:00 | 521.10 | 520.14 | 520.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 521.00 | 520.31 | 520.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:15:00 | 523.00 | 520.31 | 520.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 520.00 | 520.30 | 520.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:15:00 | 521.05 | 520.30 | 520.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 11:15:00 | 520.50 | 520.34 | 520.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:30:00 | 521.35 | 520.34 | 520.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 522.00 | 520.67 | 520.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:00:00 | 522.00 | 520.67 | 520.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 521.05 | 520.75 | 520.79 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 14:15:00 | 521.45 | 520.89 | 520.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 15:15:00 | 522.40 | 521.19 | 520.99 | Break + close above crossover candle high |

### Cycle 77 — SELL (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 09:15:00 | 518.30 | 520.61 | 520.75 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 12:15:00 | 523.90 | 521.32 | 521.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 13:15:00 | 524.35 | 521.92 | 521.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 09:15:00 | 529.50 | 529.91 | 527.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 09:30:00 | 529.00 | 529.91 | 527.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 526.15 | 529.71 | 528.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:00:00 | 526.15 | 529.71 | 528.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 528.35 | 529.43 | 528.75 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 12:15:00 | 514.10 | 525.80 | 527.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 503.50 | 521.34 | 525.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 506.40 | 505.49 | 509.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-09 09:45:00 | 506.00 | 505.49 | 509.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 505.50 | 505.03 | 506.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:30:00 | 506.20 | 505.03 | 506.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 506.25 | 505.27 | 506.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 15:00:00 | 506.25 | 505.27 | 506.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 506.00 | 505.42 | 506.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 505.90 | 505.42 | 506.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 504.90 | 505.31 | 506.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:00:00 | 503.80 | 504.99 | 505.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 504.00 | 500.12 | 500.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 13:00:00 | 502.95 | 500.68 | 500.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 13:15:00 | 504.35 | 501.42 | 501.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 504.35 | 501.42 | 501.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 11:15:00 | 506.15 | 503.48 | 502.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 504.10 | 504.48 | 503.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 14:15:00 | 504.10 | 504.48 | 503.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 504.10 | 504.48 | 503.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:45:00 | 502.50 | 504.48 | 503.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 498.95 | 503.46 | 502.97 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 498.15 | 502.40 | 502.53 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 505.10 | 502.80 | 502.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 11:15:00 | 506.00 | 504.08 | 503.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 507.55 | 507.83 | 506.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 15:00:00 | 507.55 | 507.83 | 506.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 507.55 | 508.81 | 507.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 507.75 | 508.81 | 507.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 506.65 | 508.38 | 507.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 505.40 | 508.38 | 507.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 505.35 | 507.77 | 507.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:30:00 | 505.50 | 507.77 | 507.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 12:15:00 | 505.30 | 507.28 | 507.34 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 510.15 | 507.61 | 507.41 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 13:15:00 | 506.05 | 508.02 | 508.08 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 510.60 | 508.24 | 508.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 13:15:00 | 514.85 | 510.53 | 509.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 10:15:00 | 527.20 | 527.34 | 521.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 11:00:00 | 527.20 | 527.34 | 521.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 552.50 | 557.10 | 552.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 552.50 | 557.10 | 552.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 552.20 | 556.12 | 552.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:30:00 | 551.20 | 556.12 | 552.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 553.50 | 555.60 | 552.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 548.35 | 555.60 | 552.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 549.95 | 554.47 | 552.49 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 14:15:00 | 550.35 | 551.51 | 551.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 544.85 | 550.09 | 550.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 11:15:00 | 551.30 | 550.32 | 550.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 11:15:00 | 551.30 | 550.32 | 550.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 551.30 | 550.32 | 550.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 551.30 | 550.32 | 550.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 551.40 | 550.53 | 550.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:30:00 | 551.90 | 550.53 | 550.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 551.90 | 550.81 | 551.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 551.90 | 550.81 | 551.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 550.45 | 550.73 | 550.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 15:15:00 | 549.00 | 550.73 | 550.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 557.00 | 551.71 | 551.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 557.00 | 551.71 | 551.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 559.50 | 554.62 | 552.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 14:15:00 | 554.65 | 555.51 | 553.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 15:00:00 | 554.65 | 555.51 | 553.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 554.30 | 555.26 | 553.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 549.90 | 555.26 | 553.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 550.00 | 554.21 | 553.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:15:00 | 548.20 | 554.21 | 553.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 545.95 | 552.56 | 552.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 11:15:00 | 544.40 | 550.93 | 551.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 542.20 | 541.33 | 545.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 542.20 | 541.33 | 545.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 537.05 | 537.64 | 540.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 539.50 | 538.12 | 540.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 539.00 | 538.30 | 540.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 536.30 | 538.30 | 540.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 15:15:00 | 541.00 | 537.98 | 539.02 | SL hit (close>static) qty=1.00 sl=540.70 alert=retest2 |

### Cycle 90 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 550.50 | 540.48 | 540.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 557.45 | 552.36 | 547.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 558.65 | 559.14 | 555.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 558.65 | 559.14 | 555.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 563.55 | 567.32 | 565.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 563.55 | 567.32 | 565.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 563.85 | 566.62 | 564.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:15:00 | 562.10 | 566.62 | 564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 559.65 | 565.23 | 564.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 559.65 | 565.23 | 564.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 559.95 | 563.36 | 563.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 554.65 | 561.62 | 562.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 15:15:00 | 553.00 | 550.03 | 553.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 553.00 | 550.03 | 553.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 553.00 | 550.03 | 553.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 551.75 | 550.03 | 553.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 548.95 | 549.81 | 552.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 546.75 | 549.35 | 551.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:30:00 | 546.10 | 548.45 | 551.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 563.60 | 551.01 | 551.78 | SL hit (close>static) qty=1.00 sl=561.20 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 11:15:00 | 559.75 | 552.99 | 552.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 577.15 | 560.12 | 556.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 585.50 | 595.64 | 583.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 585.50 | 595.64 | 583.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 585.50 | 595.64 | 583.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 585.55 | 595.64 | 583.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 614.95 | 616.73 | 610.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:30:00 | 612.95 | 616.73 | 610.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 615.45 | 615.58 | 612.34 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 611.50 | 613.67 | 613.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 604.00 | 608.98 | 610.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 12:15:00 | 601.80 | 600.80 | 603.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 13:00:00 | 601.80 | 600.80 | 603.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 601.55 | 600.03 | 602.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 603.40 | 600.03 | 602.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 603.55 | 600.73 | 602.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:30:00 | 599.70 | 601.58 | 602.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:30:00 | 598.60 | 601.16 | 602.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 599.65 | 601.16 | 602.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 599.10 | 598.62 | 598.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 594.40 | 596.81 | 597.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 593.55 | 596.06 | 597.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 592.50 | 595.65 | 597.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 600.50 | 596.97 | 597.44 | SL hit (close>static) qty=1.00 sl=598.50 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 601.55 | 598.15 | 597.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 602.15 | 599.44 | 598.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 598.60 | 600.81 | 599.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 598.60 | 600.81 | 599.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 598.60 | 600.81 | 599.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 598.60 | 600.81 | 599.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 596.70 | 599.99 | 599.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:30:00 | 596.45 | 599.99 | 599.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 601.10 | 600.06 | 599.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 603.30 | 600.06 | 599.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 607.45 | 601.54 | 600.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:30:00 | 610.40 | 604.16 | 602.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:00:00 | 610.60 | 604.16 | 602.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:45:00 | 610.25 | 605.33 | 603.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 13:15:00 | 609.80 | 606.15 | 603.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 604.10 | 607.60 | 606.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 604.10 | 607.60 | 606.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 604.75 | 607.03 | 605.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 606.80 | 605.60 | 605.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 606.50 | 608.17 | 607.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 604.45 | 607.42 | 607.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 604.45 | 607.42 | 607.54 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 628.45 | 611.05 | 608.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 634.45 | 629.89 | 625.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 15:15:00 | 630.00 | 631.06 | 626.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 625.95 | 631.06 | 626.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 626.70 | 630.19 | 626.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 627.40 | 630.19 | 626.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 627.40 | 629.63 | 626.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 627.95 | 629.63 | 626.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 629.25 | 629.56 | 627.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:15:00 | 630.00 | 629.23 | 627.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 630.80 | 629.57 | 628.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 11:45:00 | 630.10 | 629.62 | 628.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 630.20 | 629.90 | 628.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 630.95 | 630.52 | 629.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 633.25 | 630.52 | 629.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:45:00 | 632.20 | 632.24 | 631.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 633.15 | 638.59 | 639.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 633.15 | 638.59 | 639.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 631.65 | 637.20 | 638.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 637.65 | 637.29 | 638.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 637.65 | 637.29 | 638.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 637.65 | 637.29 | 638.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 636.35 | 637.29 | 638.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 637.15 | 637.26 | 638.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 637.15 | 637.26 | 638.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 636.25 | 637.06 | 638.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 637.55 | 637.06 | 638.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 637.30 | 636.20 | 637.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 636.75 | 636.20 | 637.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 637.10 | 636.38 | 637.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 635.30 | 636.38 | 637.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 633.90 | 635.88 | 636.97 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 645.95 | 637.90 | 637.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 648.65 | 640.05 | 638.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 627.50 | 644.06 | 642.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 627.50 | 644.06 | 642.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 627.50 | 644.06 | 642.15 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 11:15:00 | 632.00 | 639.61 | 640.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 10:15:00 | 627.90 | 634.24 | 637.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 13:15:00 | 633.85 | 632.54 | 635.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 14:00:00 | 633.85 | 632.54 | 635.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 631.60 | 631.86 | 634.31 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 11:15:00 | 638.55 | 634.24 | 634.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 13:15:00 | 641.25 | 636.36 | 635.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 639.35 | 639.45 | 637.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 11:45:00 | 640.05 | 639.45 | 637.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 638.00 | 639.26 | 637.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 638.00 | 639.26 | 637.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 634.15 | 638.24 | 637.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 634.15 | 638.24 | 637.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 634.90 | 637.57 | 637.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 631.80 | 637.57 | 637.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 637.35 | 637.74 | 637.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:30:00 | 636.95 | 637.74 | 637.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 635.60 | 637.31 | 637.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:00:00 | 635.60 | 637.31 | 637.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 634.85 | 636.82 | 636.92 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 648.50 | 638.99 | 637.85 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 631.20 | 637.71 | 638.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 626.70 | 634.28 | 636.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 09:15:00 | 638.60 | 633.98 | 636.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 638.60 | 633.98 | 636.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 638.60 | 633.98 | 636.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 10:00:00 | 638.60 | 633.98 | 636.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 634.95 | 634.17 | 635.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 10:30:00 | 636.70 | 634.17 | 635.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 634.90 | 632.48 | 634.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:00:00 | 634.90 | 632.48 | 634.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 634.05 | 632.79 | 634.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:30:00 | 633.55 | 632.79 | 634.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 636.25 | 633.52 | 634.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 638.35 | 633.52 | 634.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 632.45 | 633.30 | 634.40 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 638.95 | 635.38 | 635.21 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 633.85 | 635.07 | 635.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 628.15 | 633.69 | 634.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 635.50 | 633.22 | 634.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 635.50 | 633.22 | 634.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 635.50 | 633.22 | 634.06 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 638.75 | 635.00 | 634.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 639.65 | 636.61 | 635.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 637.60 | 637.93 | 636.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:15:00 | 636.65 | 637.93 | 636.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 636.65 | 637.68 | 636.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 638.90 | 637.68 | 636.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 634.60 | 637.01 | 636.73 | SL hit (close<static) qty=1.00 sl=635.80 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 630.00 | 635.61 | 636.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 626.35 | 633.76 | 635.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 611.65 | 607.31 | 611.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 611.65 | 607.31 | 611.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 611.65 | 607.31 | 611.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 610.60 | 607.31 | 611.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 612.85 | 608.42 | 612.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 613.35 | 608.42 | 612.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 611.60 | 609.05 | 611.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:30:00 | 611.90 | 609.05 | 611.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 612.75 | 609.79 | 612.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 613.40 | 609.79 | 612.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 615.00 | 610.83 | 612.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 614.70 | 610.83 | 612.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 617.20 | 613.29 | 613.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 619.70 | 614.57 | 613.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 619.40 | 620.28 | 617.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 619.40 | 620.28 | 617.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 617.80 | 619.78 | 617.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 618.55 | 619.78 | 617.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 620.75 | 619.97 | 618.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 13:15:00 | 622.10 | 619.97 | 618.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 623.45 | 621.17 | 619.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 639.55 | 644.31 | 644.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 639.55 | 644.31 | 644.41 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 648.05 | 644.25 | 644.23 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 643.85 | 644.17 | 644.20 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 15:15:00 | 646.05 | 644.54 | 644.36 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 642.40 | 644.12 | 644.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 639.45 | 643.18 | 643.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 645.00 | 643.40 | 643.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 12:15:00 | 645.00 | 643.40 | 643.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 645.00 | 643.40 | 643.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 647.90 | 643.40 | 643.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 640.90 | 642.90 | 643.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 640.60 | 642.95 | 643.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 639.65 | 642.49 | 643.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 639.95 | 641.90 | 642.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:00:00 | 638.00 | 640.02 | 641.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 642.70 | 638.55 | 639.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 641.05 | 638.55 | 639.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 646.65 | 640.17 | 640.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 646.65 | 640.17 | 640.34 | SL hit (close>static) qty=1.00 sl=645.30 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 644.55 | 641.05 | 640.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 12:15:00 | 647.90 | 643.00 | 641.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 647.80 | 648.29 | 645.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:00:00 | 647.80 | 648.29 | 645.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 645.30 | 647.69 | 645.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 645.60 | 647.69 | 645.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 644.95 | 647.14 | 645.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 647.45 | 647.14 | 645.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 14:30:00 | 646.00 | 647.59 | 646.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 651.30 | 646.88 | 646.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 657.90 | 662.70 | 663.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 657.90 | 662.70 | 663.12 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 665.70 | 663.02 | 662.92 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 661.85 | 662.79 | 662.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 12:15:00 | 658.90 | 662.01 | 662.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 661.55 | 661.19 | 661.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 11:15:00 | 661.55 | 661.19 | 661.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 661.55 | 661.19 | 661.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 661.55 | 661.19 | 661.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 662.00 | 661.36 | 661.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:30:00 | 662.05 | 661.36 | 661.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 662.00 | 661.48 | 661.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 664.20 | 661.48 | 661.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 660.50 | 661.29 | 661.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:00:00 | 655.85 | 659.43 | 660.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:15:00 | 656.30 | 658.93 | 660.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 664.30 | 659.47 | 659.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 664.30 | 659.47 | 659.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 666.00 | 660.77 | 659.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 664.05 | 665.60 | 663.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 11:15:00 | 664.05 | 665.60 | 663.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 664.05 | 665.60 | 663.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 664.05 | 665.60 | 663.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 663.80 | 665.24 | 663.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 663.20 | 665.24 | 663.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 663.90 | 664.97 | 663.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 663.90 | 664.97 | 663.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 661.05 | 664.19 | 663.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:45:00 | 660.05 | 664.19 | 663.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 662.40 | 663.83 | 663.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 665.55 | 663.83 | 663.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 655.50 | 662.17 | 662.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 653.75 | 659.42 | 661.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 629.65 | 627.42 | 634.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:45:00 | 627.90 | 627.42 | 634.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 633.35 | 629.65 | 632.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 633.35 | 629.65 | 632.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 632.00 | 630.12 | 632.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 633.45 | 630.12 | 632.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 633.80 | 630.86 | 632.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 634.20 | 630.86 | 632.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 632.05 | 631.09 | 632.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:45:00 | 629.95 | 631.06 | 632.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:30:00 | 630.00 | 630.48 | 632.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 598.45 | 613.52 | 621.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 598.50 | 613.52 | 621.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 10:15:00 | 566.96 | 573.99 | 586.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 572.80 | 571.22 | 571.12 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 570.75 | 571.10 | 571.11 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 571.60 | 571.20 | 571.15 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 568.75 | 570.71 | 570.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 567.25 | 569.20 | 570.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 570.45 | 568.54 | 569.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 570.45 | 568.54 | 569.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 570.45 | 568.54 | 569.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 570.45 | 568.54 | 569.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 571.55 | 569.14 | 569.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 571.40 | 569.14 | 569.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 571.25 | 569.68 | 569.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 571.25 | 569.68 | 569.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 570.95 | 569.94 | 569.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 573.40 | 570.76 | 570.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 573.60 | 575.13 | 573.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 573.60 | 575.13 | 573.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 573.60 | 575.13 | 573.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 573.60 | 575.13 | 573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 577.65 | 575.63 | 573.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 572.70 | 575.63 | 573.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 575.20 | 575.55 | 573.74 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 567.10 | 572.00 | 572.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 563.85 | 567.57 | 569.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 547.40 | 542.00 | 546.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 547.40 | 542.00 | 546.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 547.40 | 542.00 | 546.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 547.40 | 542.00 | 546.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 553.35 | 544.27 | 547.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 553.35 | 544.27 | 547.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 553.40 | 546.10 | 547.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:30:00 | 552.85 | 546.10 | 547.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 556.30 | 549.74 | 549.08 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 536.65 | 546.58 | 547.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 534.25 | 542.26 | 545.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 542.00 | 539.63 | 543.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 542.00 | 539.63 | 543.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 542.00 | 539.63 | 543.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 542.00 | 539.63 | 543.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 541.40 | 540.11 | 542.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:30:00 | 542.40 | 540.11 | 542.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 541.75 | 540.44 | 542.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:45:00 | 543.30 | 540.44 | 542.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 541.65 | 540.68 | 542.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 541.65 | 540.68 | 542.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 547.30 | 542.00 | 542.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:45:00 | 547.55 | 542.00 | 542.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 546.00 | 542.80 | 543.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 540.75 | 542.68 | 543.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 540.90 | 542.15 | 542.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 539.90 | 541.82 | 542.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 539.35 | 542.42 | 542.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 536.10 | 541.16 | 542.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 534.45 | 541.16 | 542.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 534.90 | 536.76 | 539.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 534.15 | 536.38 | 538.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 534.90 | 533.61 | 535.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 538.55 | 534.81 | 536.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 538.55 | 534.81 | 536.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 539.65 | 535.77 | 536.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 540.05 | 535.77 | 536.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 539.35 | 537.29 | 537.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 539.35 | 537.29 | 537.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 15:15:00 | 540.20 | 537.88 | 537.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 534.40 | 537.18 | 537.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 534.40 | 537.18 | 537.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 534.40 | 537.18 | 537.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 534.40 | 537.18 | 537.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 534.05 | 536.55 | 536.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 532.70 | 535.78 | 536.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 12:15:00 | 536.20 | 535.87 | 536.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 12:15:00 | 536.20 | 535.87 | 536.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 536.20 | 535.87 | 536.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:45:00 | 536.95 | 535.87 | 536.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 534.85 | 535.66 | 536.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:30:00 | 533.50 | 535.40 | 536.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 531.00 | 535.14 | 535.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 10:15:00 | 506.82 | 510.79 | 515.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 12:15:00 | 504.45 | 508.36 | 513.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 14:15:00 | 508.45 | 507.88 | 512.36 | SL hit (close>ema200) qty=0.50 sl=507.88 alert=retest2 |

### Cycle 130 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 512.75 | 510.32 | 510.31 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 507.35 | 510.21 | 510.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 504.25 | 508.60 | 509.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 507.75 | 507.16 | 508.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 507.75 | 507.16 | 508.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 507.75 | 507.16 | 508.17 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 513.20 | 509.33 | 508.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 516.95 | 511.97 | 510.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 522.90 | 524.22 | 520.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 522.90 | 524.22 | 520.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 522.90 | 524.22 | 520.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:30:00 | 526.60 | 524.74 | 522.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 526.95 | 524.74 | 522.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 10:45:00 | 528.35 | 526.49 | 523.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 12:00:00 | 527.30 | 526.65 | 524.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 525.15 | 526.45 | 524.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 525.15 | 526.45 | 524.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 526.15 | 526.39 | 524.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 526.15 | 526.39 | 524.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 525.70 | 526.25 | 524.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 524.50 | 526.25 | 524.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 523.70 | 525.74 | 524.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 527.85 | 526.06 | 525.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 522.10 | 524.46 | 524.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 522.10 | 524.46 | 524.70 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 526.50 | 524.87 | 524.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 527.50 | 525.40 | 525.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 14:15:00 | 522.25 | 525.07 | 524.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 14:15:00 | 522.25 | 525.07 | 524.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 522.25 | 525.07 | 524.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 522.25 | 525.07 | 524.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 523.30 | 524.72 | 524.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 520.55 | 523.86 | 524.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 521.50 | 521.19 | 522.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 12:15:00 | 521.50 | 521.19 | 522.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 521.50 | 521.19 | 522.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 520.10 | 521.90 | 522.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 524.95 | 522.51 | 522.74 | SL hit (close>static) qty=1.00 sl=523.70 alert=retest2 |

### Cycle 136 — BUY (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 15:15:00 | 523.50 | 522.81 | 522.74 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 504.50 | 519.15 | 521.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 500.90 | 505.80 | 507.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 508.80 | 505.93 | 507.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 508.80 | 505.93 | 507.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 508.80 | 505.93 | 507.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 508.80 | 505.93 | 507.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 508.65 | 506.48 | 507.51 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 511.95 | 508.18 | 508.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 515.00 | 510.15 | 509.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 13:15:00 | 510.70 | 510.78 | 509.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:45:00 | 510.25 | 510.78 | 509.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 509.85 | 510.59 | 509.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:30:00 | 509.75 | 510.59 | 509.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 510.50 | 510.57 | 509.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 507.70 | 510.57 | 509.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 506.15 | 509.69 | 509.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 506.15 | 509.69 | 509.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 507.80 | 509.31 | 509.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 504.95 | 507.68 | 508.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 15:15:00 | 505.80 | 505.77 | 506.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 09:15:00 | 502.50 | 505.77 | 506.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 503.25 | 503.70 | 505.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 504.50 | 503.70 | 505.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 504.95 | 504.21 | 505.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 507.40 | 504.21 | 505.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 507.25 | 504.82 | 505.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 507.25 | 504.82 | 505.29 | SL hit (close>ema400) qty=1.00 sl=505.29 alert=retest1 |

### Cycle 140 — BUY (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 13:15:00 | 506.90 | 505.69 | 505.62 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 501.05 | 504.76 | 505.20 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2024-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 12:15:00 | 507.05 | 505.14 | 505.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 14:15:00 | 510.05 | 506.16 | 505.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 509.50 | 509.77 | 508.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 15:15:00 | 509.50 | 509.77 | 508.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 509.50 | 509.77 | 508.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 507.50 | 509.77 | 508.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 505.50 | 508.92 | 508.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 505.50 | 508.92 | 508.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 508.05 | 508.74 | 508.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:30:00 | 505.90 | 508.74 | 508.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 506.55 | 508.31 | 507.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 506.55 | 508.31 | 507.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 507.50 | 508.14 | 507.88 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 14:15:00 | 505.40 | 507.32 | 507.53 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 508.20 | 507.60 | 507.56 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 507.25 | 507.50 | 507.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 505.30 | 507.06 | 507.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 10:15:00 | 508.30 | 507.31 | 507.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 10:15:00 | 508.30 | 507.31 | 507.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 508.30 | 507.31 | 507.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:30:00 | 508.65 | 507.31 | 507.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 507.00 | 507.25 | 507.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:30:00 | 508.30 | 507.25 | 507.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 508.05 | 507.41 | 507.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:45:00 | 509.30 | 507.41 | 507.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 506.20 | 507.16 | 507.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 14:15:00 | 505.85 | 507.16 | 507.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 503.70 | 506.47 | 507.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 505.40 | 506.66 | 507.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 507.70 | 506.89 | 506.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 507.70 | 506.89 | 506.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 510.50 | 507.91 | 507.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 505.45 | 518.00 | 515.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 505.45 | 518.00 | 515.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 505.45 | 518.00 | 515.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 502.20 | 518.00 | 515.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 502.70 | 514.94 | 514.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 502.70 | 514.94 | 514.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 507.00 | 513.35 | 513.63 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 513.00 | 510.60 | 510.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 513.95 | 511.58 | 511.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 516.60 | 518.97 | 516.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 516.60 | 518.97 | 516.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 516.60 | 518.97 | 516.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 516.60 | 518.97 | 516.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 520.60 | 519.30 | 516.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 518.30 | 519.30 | 516.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 516.50 | 519.01 | 517.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 516.50 | 519.01 | 517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 517.45 | 518.70 | 517.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 12:15:00 | 518.45 | 516.81 | 516.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 12:15:00 | 515.50 | 516.55 | 516.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 515.50 | 516.55 | 516.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 511.85 | 515.61 | 516.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 516.45 | 514.42 | 515.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 516.45 | 514.42 | 515.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 516.45 | 514.42 | 515.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 516.40 | 514.42 | 515.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 519.30 | 515.40 | 515.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 519.30 | 515.40 | 515.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 516.70 | 515.66 | 515.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 515.20 | 515.66 | 515.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 517.30 | 515.19 | 514.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 517.30 | 515.19 | 514.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 521.75 | 516.79 | 515.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 523.45 | 525.53 | 522.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 523.45 | 525.53 | 522.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 523.45 | 525.53 | 522.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 522.05 | 525.53 | 522.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 524.90 | 524.96 | 523.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 521.70 | 524.96 | 523.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 521.50 | 524.27 | 523.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 521.50 | 524.27 | 523.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 521.50 | 523.72 | 522.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 524.70 | 523.72 | 522.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:30:00 | 522.40 | 523.52 | 522.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 15:15:00 | 520.80 | 523.21 | 523.15 | SL hit (close<static) qty=1.00 sl=520.95 alert=retest2 |

### Cycle 151 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 518.00 | 522.17 | 522.69 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 524.55 | 522.25 | 522.17 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 522.35 | 522.91 | 522.95 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 09:15:00 | 528.80 | 524.09 | 523.48 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 09:15:00 | 516.10 | 523.34 | 524.03 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 534.70 | 523.87 | 522.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 538.10 | 526.72 | 524.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 529.95 | 529.97 | 526.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 529.95 | 529.97 | 526.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 535.40 | 531.58 | 528.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 523.85 | 531.58 | 528.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 538.45 | 537.83 | 533.69 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 531.35 | 536.36 | 536.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 530.10 | 533.62 | 535.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 527.00 | 526.76 | 529.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 14:30:00 | 526.70 | 526.76 | 529.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 531.80 | 527.80 | 529.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 531.80 | 527.80 | 529.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 530.20 | 528.28 | 529.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:30:00 | 529.70 | 528.13 | 529.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 10:15:00 | 525.60 | 524.61 | 524.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 525.60 | 524.61 | 524.58 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 522.15 | 524.18 | 524.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 14:15:00 | 519.70 | 523.29 | 523.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 09:15:00 | 523.30 | 522.67 | 523.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 523.30 | 522.67 | 523.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 523.30 | 522.67 | 523.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 523.30 | 522.67 | 523.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 520.80 | 522.30 | 523.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 518.50 | 522.30 | 523.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:30:00 | 519.15 | 521.18 | 522.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 13:30:00 | 519.40 | 520.71 | 522.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:45:00 | 519.00 | 520.62 | 522.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 517.20 | 519.88 | 521.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 515.50 | 519.00 | 520.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 514.35 | 518.78 | 519.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 516.00 | 517.61 | 519.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 11:15:00 | 511.25 | 507.88 | 507.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 511.25 | 507.88 | 507.76 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 502.70 | 507.08 | 507.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 494.60 | 502.60 | 505.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 492.50 | 491.77 | 495.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 09:15:00 | 487.00 | 491.77 | 495.25 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 490.05 | 486.44 | 489.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 490.05 | 486.44 | 489.08 | SL hit (close>ema400) qty=1.00 sl=489.08 alert=retest1 |

### Cycle 162 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 495.70 | 489.94 | 489.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 11:15:00 | 497.15 | 491.38 | 490.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 493.70 | 494.10 | 492.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 493.70 | 494.10 | 492.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 493.70 | 494.10 | 492.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 493.70 | 494.10 | 492.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 500.30 | 495.83 | 494.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 501.85 | 498.97 | 497.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 501.50 | 499.50 | 497.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 492.50 | 498.16 | 498.13 | SL hit (close<static) qty=1.00 sl=492.60 alert=retest2 |

### Cycle 163 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 492.15 | 496.96 | 497.59 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 499.65 | 497.13 | 497.11 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 11:15:00 | 495.25 | 497.19 | 497.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 13:15:00 | 493.20 | 496.14 | 496.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 496.60 | 495.91 | 496.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 10:15:00 | 496.60 | 495.91 | 496.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 496.60 | 495.91 | 496.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:30:00 | 496.10 | 495.91 | 496.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 496.90 | 496.11 | 496.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:45:00 | 496.10 | 496.11 | 496.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 496.75 | 496.23 | 496.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:45:00 | 496.65 | 496.23 | 496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 498.25 | 496.64 | 496.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:30:00 | 497.65 | 496.64 | 496.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 499.30 | 497.17 | 496.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 504.70 | 498.74 | 497.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 10:15:00 | 502.40 | 504.15 | 501.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:00:00 | 502.40 | 504.15 | 501.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 507.40 | 508.53 | 506.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 13:30:00 | 509.60 | 508.67 | 507.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 509.35 | 508.75 | 507.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 11:45:00 | 509.45 | 510.90 | 509.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 14:15:00 | 506.45 | 508.99 | 509.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 506.45 | 508.99 | 509.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 505.70 | 508.33 | 508.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 509.15 | 508.49 | 508.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 509.15 | 508.49 | 508.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 509.15 | 508.49 | 508.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 510.00 | 508.49 | 508.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 503.15 | 507.43 | 508.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 501.75 | 507.43 | 508.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 476.66 | 489.38 | 496.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 451.57 | 460.68 | 470.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 168 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 467.10 | 462.91 | 462.35 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 13:15:00 | 461.05 | 462.39 | 462.50 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 464.90 | 462.85 | 462.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 467.40 | 463.76 | 463.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 477.20 | 478.53 | 475.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 10:00:00 | 477.20 | 478.53 | 475.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 475.00 | 477.49 | 476.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:45:00 | 475.25 | 477.49 | 476.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 476.20 | 477.23 | 476.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 477.30 | 476.64 | 476.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 11:15:00 | 483.90 | 484.92 | 484.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 11:15:00 | 483.90 | 484.92 | 484.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 12:15:00 | 482.80 | 484.49 | 484.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 483.75 | 483.52 | 484.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 483.75 | 483.52 | 484.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 483.75 | 483.52 | 484.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 485.65 | 483.52 | 484.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 483.40 | 483.50 | 484.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 484.00 | 483.50 | 484.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 483.10 | 483.42 | 483.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:30:00 | 483.80 | 483.42 | 483.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 483.65 | 483.47 | 483.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:30:00 | 483.75 | 483.47 | 483.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 483.45 | 483.46 | 483.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 483.85 | 483.46 | 483.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 483.95 | 483.56 | 483.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 488.45 | 483.56 | 483.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 486.15 | 484.08 | 484.08 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 479.50 | 484.58 | 484.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 477.75 | 483.22 | 484.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 482.55 | 482.02 | 483.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 482.55 | 482.02 | 483.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 482.55 | 482.02 | 483.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 485.15 | 482.02 | 483.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 481.70 | 481.96 | 483.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 483.45 | 481.96 | 483.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 481.50 | 481.54 | 482.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 481.50 | 481.54 | 482.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 484.20 | 482.14 | 482.59 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 484.05 | 482.95 | 482.91 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 482.40 | 482.79 | 482.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 480.70 | 482.37 | 482.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 482.80 | 481.50 | 482.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 482.80 | 481.50 | 482.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 482.80 | 481.50 | 482.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 482.80 | 481.50 | 482.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 482.25 | 481.65 | 482.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:15:00 | 484.50 | 481.65 | 482.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 483.70 | 482.06 | 482.26 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 484.75 | 482.60 | 482.49 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 15:15:00 | 480.00 | 482.28 | 482.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 09:15:00 | 474.95 | 480.81 | 481.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 472.45 | 465.85 | 469.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 472.45 | 465.85 | 469.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 472.45 | 465.85 | 469.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 472.45 | 465.85 | 469.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 472.45 | 467.17 | 469.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:30:00 | 472.70 | 467.17 | 469.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 475.90 | 471.55 | 471.15 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 469.70 | 471.63 | 471.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 467.90 | 470.88 | 471.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 12:15:00 | 468.15 | 470.20 | 470.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 468.55 | 469.53 | 470.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 468.05 | 469.13 | 469.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 10:15:00 | 473.05 | 469.92 | 470.22 | SL hit (close>static) qty=1.00 sl=472.50 alert=retest2 |

### Cycle 180 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 473.20 | 470.57 | 470.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 474.30 | 471.32 | 470.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 480.65 | 481.84 | 478.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:45:00 | 480.40 | 481.84 | 478.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 479.35 | 481.34 | 479.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 479.35 | 481.34 | 479.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 475.75 | 480.22 | 478.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 475.75 | 480.22 | 478.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 476.10 | 479.40 | 478.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 480.10 | 479.40 | 478.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 476.30 | 479.38 | 479.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 476.30 | 479.38 | 479.60 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 483.00 | 480.16 | 479.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 483.90 | 482.80 | 482.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 482.30 | 483.88 | 483.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 483.35 | 483.77 | 483.17 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 481.10 | 482.82 | 482.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 479.95 | 481.83 | 482.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 481.40 | 480.95 | 481.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 481.75 | 481.11 | 481.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 481.75 | 481.11 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 483.15 | 481.52 | 481.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 487.30 | 481.52 | 481.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 486.45 | 482.50 | 482.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 489.50 | 483.90 | 482.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 482.25 | 483.85 | 483.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 484.25 | 483.93 | 483.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 485.10 | 484.05 | 483.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 13:00:00 | 485.50 | 484.46 | 483.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 490.30 | 484.90 | 484.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 483.25 | 486.66 | 487.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 483.25 | 486.66 | 487.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 482.45 | 485.82 | 486.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 484.00 | 483.55 | 485.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:45:00 | 484.15 | 483.55 | 485.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 485.10 | 484.04 | 484.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 485.10 | 484.04 | 484.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 485.70 | 484.37 | 484.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 485.70 | 484.37 | 484.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 486.50 | 485.31 | 485.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 488.00 | 486.13 | 485.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 489.10 | 490.01 | 488.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 488.15 | 489.64 | 488.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 488.15 | 489.64 | 488.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 486.05 | 488.92 | 488.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 484.95 | 488.92 | 488.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 481.15 | 487.37 | 487.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 479.15 | 485.72 | 486.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 469.50 | 469.10 | 473.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 469.50 | 469.10 | 473.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 473.20 | 470.50 | 473.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 473.20 | 470.50 | 473.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 471.95 | 470.79 | 473.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 470.65 | 471.26 | 472.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 470.55 | 471.43 | 472.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 469.75 | 471.23 | 472.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:30:00 | 470.25 | 470.63 | 471.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 469.90 | 470.49 | 471.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 469.90 | 470.49 | 471.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 467.35 | 466.85 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 467.35 | 466.85 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 466.90 | 466.35 | 467.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 466.90 | 466.35 | 467.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 467.70 | 466.62 | 467.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 467.70 | 466.62 | 467.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 467.95 | 466.88 | 467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 467.95 | 466.88 | 467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 467.50 | 467.01 | 467.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 471.95 | 467.01 | 467.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 470.65 | 467.74 | 467.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 470.65 | 467.74 | 467.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 473.25 | 469.44 | 468.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 480.70 | 483.98 | 481.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 481.10 | 483.40 | 481.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 481.00 | 483.40 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 483.45 | 484.69 | 483.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 483.45 | 484.69 | 483.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 485.10 | 484.77 | 483.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 486.50 | 484.18 | 483.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 486.00 | 485.15 | 484.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-11 15:15:00 | 535.15 | 527.26 | 522.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 520.55 | 527.00 | 527.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 518.50 | 521.30 | 523.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 514.50 | 514.40 | 517.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 514.50 | 514.40 | 517.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 512.25 | 514.22 | 516.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:30:00 | 510.80 | 511.88 | 513.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 520.80 | 514.19 | 513.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 520.80 | 514.19 | 513.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 11:15:00 | 521.35 | 515.62 | 514.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 518.60 | 518.86 | 516.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 518.55 | 518.86 | 516.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 515.20 | 518.12 | 516.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:45:00 | 515.65 | 518.12 | 516.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 516.70 | 517.84 | 516.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:00:00 | 517.70 | 517.70 | 516.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 517.70 | 517.70 | 516.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 521.80 | 529.82 | 530.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 521.80 | 529.82 | 530.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 520.30 | 524.55 | 526.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 517.25 | 515.87 | 518.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 517.25 | 515.87 | 518.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 517.60 | 516.22 | 518.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 516.65 | 516.22 | 518.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 511.45 | 515.26 | 518.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 510.35 | 513.75 | 516.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:30:00 | 510.50 | 512.36 | 515.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 525.50 | 506.32 | 505.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 525.50 | 506.32 | 505.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 532.00 | 525.11 | 520.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 527.90 | 529.39 | 524.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 525.85 | 529.39 | 524.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 524.00 | 528.31 | 524.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 524.85 | 528.31 | 524.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 524.30 | 527.51 | 524.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:15:00 | 524.05 | 527.51 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 524.10 | 526.83 | 524.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 523.60 | 526.83 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 521.95 | 525.85 | 523.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 521.95 | 525.85 | 523.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 518.60 | 524.40 | 523.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 518.60 | 524.40 | 523.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 518.70 | 522.07 | 522.44 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 524.05 | 520.23 | 520.03 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 517.40 | 520.49 | 520.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 515.40 | 519.48 | 520.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 518.20 | 516.64 | 518.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 518.20 | 516.64 | 518.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 520.55 | 517.42 | 518.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 520.55 | 517.42 | 518.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 521.00 | 518.14 | 518.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 521.65 | 518.14 | 518.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 520.65 | 519.17 | 519.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 524.15 | 521.96 | 520.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 546.20 | 550.43 | 546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 543.50 | 549.04 | 545.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 543.50 | 549.04 | 545.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 545.75 | 548.39 | 545.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 546.50 | 548.39 | 545.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 547.75 | 548.31 | 546.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 541.25 | 545.89 | 546.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 541.25 | 545.89 | 546.14 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 546.60 | 545.57 | 545.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 547.65 | 545.98 | 545.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 545.35 | 545.85 | 545.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 543.50 | 545.38 | 545.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 537.35 | 543.40 | 544.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 540.80 | 539.43 | 541.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 540.40 | 539.63 | 541.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 541.00 | 539.63 | 541.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 542.15 | 540.11 | 540.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 538.55 | 540.19 | 540.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 539.50 | 540.10 | 540.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 539.05 | 539.85 | 540.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 537.65 | 535.87 | 536.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 535.50 | 535.80 | 536.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 533.90 | 535.42 | 536.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 536.75 | 535.81 | 536.30 | SL hit (close>static) qty=1.00 sl=536.60 alert=retest2 |

### Cycle 200 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 489.10 | 487.54 | 487.41 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 484.30 | 487.06 | 487.30 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 487.75 | 487.04 | 487.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 491.10 | 487.85 | 487.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 506.40 | 506.60 | 502.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:45:00 | 506.55 | 506.60 | 502.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 505.50 | 505.72 | 503.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 505.50 | 505.72 | 503.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 501.30 | 508.35 | 506.81 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 503.95 | 506.38 | 506.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 502.30 | 504.81 | 505.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 507.25 | 503.75 | 504.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 504.85 | 503.97 | 504.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 506.95 | 503.97 | 504.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 507.25 | 504.63 | 505.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 507.25 | 504.63 | 505.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 507.20 | 505.14 | 505.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 507.20 | 505.14 | 505.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 507.30 | 505.57 | 505.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 508.50 | 506.56 | 505.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 503.50 | 505.95 | 505.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 504.00 | 505.56 | 505.55 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 501.00 | 504.65 | 505.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 492.15 | 500.76 | 503.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 492.25 | 491.71 | 495.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:00:00 | 492.25 | 491.71 | 495.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 500.40 | 493.15 | 495.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 500.40 | 493.15 | 495.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 503.70 | 495.26 | 496.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 503.70 | 495.26 | 496.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 503.85 | 498.10 | 497.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 511.55 | 502.46 | 499.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 523.55 | 518.81 | 517.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 522.25 | 519.50 | 517.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 522.25 | 520.25 | 518.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:45:00 | 522.25 | 520.75 | 518.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 518.25 | 520.41 | 519.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 518.25 | 520.41 | 519.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 520.95 | 520.52 | 519.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 521.85 | 521.00 | 519.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 522.05 | 522.29 | 520.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 514.15 | 522.50 | 522.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 514.15 | 522.50 | 522.68 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 525.30 | 522.09 | 521.69 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 518.75 | 521.81 | 521.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 515.85 | 519.37 | 520.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 515.15 | 513.66 | 515.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 515.15 | 513.66 | 515.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 515.00 | 513.93 | 515.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 516.50 | 513.93 | 515.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 517.35 | 514.61 | 515.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 517.35 | 514.61 | 515.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 517.55 | 515.20 | 515.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 517.55 | 515.20 | 515.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 518.15 | 516.28 | 516.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 519.25 | 517.38 | 516.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 517.30 | 517.56 | 517.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 518.50 | 517.74 | 517.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 517.85 | 517.74 | 517.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 515.75 | 517.89 | 517.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 517.55 | 517.89 | 517.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 514.00 | 517.12 | 517.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 514.00 | 517.12 | 517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 515.65 | 516.82 | 516.98 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 518.60 | 516.97 | 516.95 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 516.30 | 516.84 | 516.89 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 518.60 | 516.89 | 516.89 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 516.10 | 516.73 | 516.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 514.25 | 515.67 | 516.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 508.85 | 507.69 | 510.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 508.85 | 507.69 | 510.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 508.80 | 506.79 | 508.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 508.80 | 506.79 | 508.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 508.50 | 507.13 | 508.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 508.60 | 507.13 | 508.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 508.35 | 507.37 | 508.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 508.80 | 507.37 | 508.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 507.65 | 507.43 | 508.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 505.70 | 506.76 | 508.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 507.20 | 507.05 | 508.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 509.50 | 507.20 | 507.87 | SL hit (close>static) qty=1.00 sl=509.10 alert=retest2 |

### Cycle 216 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 510.50 | 508.35 | 508.31 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 506.85 | 508.05 | 508.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 505.15 | 507.47 | 507.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 505.00 | 504.82 | 506.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 499.50 | 504.82 | 506.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 11:30:00 | 503.50 | 504.49 | 505.65 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 504.15 | 504.42 | 505.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 505.50 | 504.42 | 505.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 503.90 | 504.32 | 505.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 504.40 | 504.32 | 505.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 504.95 | 504.23 | 505.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 504.50 | 504.23 | 505.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 505.50 | 504.49 | 505.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 505.50 | 504.49 | 505.17 | SL hit (close>ema400) qty=1.00 sl=505.17 alert=retest1 |

### Cycle 218 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 503.25 | 499.63 | 499.19 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 496.20 | 498.75 | 498.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 494.65 | 497.34 | 498.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 493.50 | 492.95 | 494.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 493.50 | 492.95 | 494.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 493.80 | 493.06 | 493.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 493.80 | 493.06 | 493.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 494.25 | 493.30 | 493.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 495.30 | 493.30 | 493.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 494.70 | 493.58 | 494.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 494.35 | 493.58 | 494.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 493.75 | 493.61 | 493.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 493.65 | 493.61 | 493.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 493.20 | 493.53 | 493.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 496.00 | 494.02 | 494.10 | SL hit (close>static) qty=1.00 sl=495.25 alert=retest2 |

### Cycle 220 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 495.45 | 494.31 | 494.22 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 493.85 | 494.13 | 494.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 493.15 | 493.86 | 494.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 494.05 | 493.89 | 494.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 492.80 | 493.67 | 493.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:15:00 | 492.70 | 493.67 | 493.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:45:00 | 492.55 | 493.10 | 493.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 492.45 | 489.74 | 489.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 494.55 | 490.70 | 490.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 494.55 | 490.70 | 490.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 505.35 | 494.87 | 492.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 501.50 | 501.81 | 498.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 501.50 | 501.81 | 498.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 498.90 | 500.77 | 498.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 498.60 | 500.77 | 498.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 499.40 | 500.50 | 498.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 501.75 | 500.12 | 498.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 518.65 | 521.41 | 521.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 518.65 | 521.41 | 521.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 515.70 | 519.50 | 520.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 510.35 | 515.42 | 516.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 511.00 | 513.02 | 514.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 511.40 | 512.66 | 514.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 517.90 | 512.77 | 512.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 517.90 | 512.77 | 512.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 525.25 | 515.27 | 513.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 524.10 | 525.75 | 521.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 524.10 | 525.75 | 521.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 520.15 | 524.63 | 521.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 520.15 | 524.63 | 521.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 518.60 | 523.42 | 521.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 520.70 | 523.42 | 521.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 515.80 | 520.48 | 520.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 515.80 | 520.48 | 520.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 515.45 | 519.48 | 519.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 513.40 | 518.26 | 519.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 513.55 | 512.23 | 515.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 513.55 | 512.23 | 515.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 515.05 | 512.46 | 514.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 515.05 | 512.46 | 514.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 517.20 | 513.41 | 514.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 509.50 | 511.81 | 514.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 514.70 | 508.99 | 510.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 514.25 | 510.41 | 511.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 508.25 | 505.02 | 503.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 517.25 | 519.39 | 516.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 517.25 | 519.39 | 516.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 518.40 | 519.19 | 516.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 520.10 | 519.27 | 516.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 520.20 | 519.60 | 517.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 512.90 | 518.25 | 517.28 | SL hit (close<static) qty=1.00 sl=516.10 alert=retest2 |

### Cycle 227 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 512.50 | 516.01 | 516.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 511.30 | 515.07 | 516.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 515.00 | 514.20 | 515.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:00:00 | 515.00 | 514.20 | 515.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 514.45 | 514.25 | 515.32 | EMA400 retest candle locked (from downside) |

### Cycle 228 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 517.05 | 515.64 | 515.56 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 506.45 | 514.12 | 514.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 505.00 | 511.46 | 513.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 510.75 | 510.69 | 512.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 510.75 | 510.69 | 512.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 512.45 | 510.98 | 512.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 512.45 | 510.98 | 512.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 508.70 | 510.53 | 512.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 507.70 | 509.60 | 511.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 507.25 | 505.89 | 508.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 515.00 | 511.60 | 510.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 512.60 | 519.86 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 516.35 | 519.16 | 518.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 517.25 | 518.93 | 518.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 496.45 | 507.32 | 511.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 488.15 | 488.01 | 494.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 485.95 | 488.01 | 494.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | SL hit (close>ema400) qty=1.00 sl=476.97 alert=retest1 |

### Cycle 232 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 437.70 | 427.71 | 427.55 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 422.15 | 427.73 | 428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 418.85 | 424.01 | 426.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 418.75 | 414.63 | 418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 415.25 | 414.75 | 418.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 414.60 | 415.48 | 417.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 413.20 | 415.31 | 417.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 414.60 | 415.31 | 417.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 419.60 | 413.82 | 414.76 | SL hit (close>static) qty=1.00 sl=419.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 424.60 | 415.89 | 414.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 428.80 | 421.66 | 418.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 424.40 | 427.97 | 428.17 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 430.00 | 428.37 | 428.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 435.70 | 430.12 | 429.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 431.90 | 432.27 | 430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 428.95 | 431.61 | 430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 428.95 | 431.61 | 430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 428.60 | 431.01 | 430.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 428.50 | 431.01 | 430.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 427.80 | 429.91 | 430.14 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 436.50 | 431.23 | 430.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 440.90 | 433.16 | 431.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 440.00 | 441.87 | 439.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 441.75 | 441.87 | 439.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 450.25 | 443.55 | 440.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 453.20 | 443.55 | 440.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 452.55 | 448.02 | 444.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | EMA200 below EMA400 |

### Cycle 240 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 455.90 | 454.10 | 453.89 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 452.00 | 453.75 | 453.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 450.10 | 452.80 | 453.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 retest candle locked (from downside) |

### Cycle 242 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 455.80 | 452.49 | 452.49 | EMA200 above EMA400 |

### Cycle 243 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 442.15 | 450.84 | 451.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 440.95 | 445.60 | 448.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 449.30 | 445.76 | 448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 449.35 | 446.48 | 448.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 449.75 | 446.48 | 448.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 446.55 | 446.80 | 448.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 444.60 | 446.80 | 448.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:45:00 | 445.00 | 446.57 | 448.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 453.45 | 447.56 | 448.06 | SL hit (close>static) qty=1.00 sl=448.90 alert=retest2 |

### Cycle 244 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 454.50 | 448.94 | 448.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 455.55 | 450.27 | 449.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 463.35 | 463.59 | 459.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 463.35 | 463.59 | 459.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 10:15:00 | 521.00 | 2023-05-17 12:15:00 | 523.55 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-05-22 13:15:00 | 520.60 | 2023-05-22 15:15:00 | 524.95 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-05-23 09:15:00 | 520.55 | 2023-05-23 10:15:00 | 523.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-05-24 15:15:00 | 528.40 | 2023-06-05 15:15:00 | 553.95 | STOP_HIT | 1.00 | 4.84% |
| BUY | retest2 | 2023-05-25 09:30:00 | 528.70 | 2023-06-05 15:15:00 | 553.95 | STOP_HIT | 1.00 | 4.78% |
| SELL | retest2 | 2023-06-07 11:15:00 | 552.95 | 2023-06-07 13:15:00 | 554.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-06-09 09:15:00 | 555.90 | 2023-06-09 09:15:00 | 550.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-06-19 14:15:00 | 569.95 | 2023-06-20 09:15:00 | 567.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-06-19 15:00:00 | 569.90 | 2023-06-20 09:15:00 | 567.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-20 11:30:00 | 570.70 | 2023-06-21 15:15:00 | 566.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-06-21 12:30:00 | 569.85 | 2023-06-21 15:15:00 | 566.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-06-27 12:15:00 | 568.90 | 2023-07-07 11:15:00 | 578.55 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest1 | 2023-07-13 09:15:00 | 585.05 | 2023-07-13 13:15:00 | 577.10 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2023-07-19 14:45:00 | 574.25 | 2023-07-20 15:15:00 | 579.75 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-07-20 09:15:00 | 573.90 | 2023-07-20 15:15:00 | 579.75 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-07-20 11:15:00 | 573.65 | 2023-07-20 15:15:00 | 579.75 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-07-21 09:15:00 | 572.55 | 2023-07-26 10:15:00 | 574.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-07-28 12:45:00 | 578.25 | 2023-08-01 10:15:00 | 573.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-08-16 09:15:00 | 561.60 | 2023-08-18 14:15:00 | 565.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-08-16 10:45:00 | 561.95 | 2023-08-18 14:15:00 | 565.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-08-17 09:30:00 | 562.60 | 2023-08-18 14:15:00 | 565.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-08-17 10:15:00 | 562.50 | 2023-08-18 14:15:00 | 565.70 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-08-24 09:15:00 | 568.75 | 2023-08-24 09:15:00 | 567.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2023-08-30 11:30:00 | 554.35 | 2023-08-31 13:15:00 | 556.85 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-08-30 15:15:00 | 554.25 | 2023-08-31 13:15:00 | 556.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-08-31 10:00:00 | 553.05 | 2023-08-31 13:15:00 | 556.85 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-08-31 14:45:00 | 554.25 | 2023-09-01 15:15:00 | 554.40 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2023-09-01 09:45:00 | 551.70 | 2023-09-01 15:15:00 | 554.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-09-01 11:30:00 | 551.65 | 2023-09-01 15:15:00 | 554.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-10-11 10:45:00 | 543.70 | 2023-10-17 14:15:00 | 541.35 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2023-10-11 14:30:00 | 543.90 | 2023-10-17 14:15:00 | 541.35 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2023-10-23 14:45:00 | 522.30 | 2023-10-27 13:15:00 | 524.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-10-25 10:00:00 | 522.80 | 2023-10-27 13:15:00 | 524.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-10-25 11:15:00 | 522.00 | 2023-10-27 13:15:00 | 524.55 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-11-13 09:15:00 | 531.60 | 2023-11-16 12:15:00 | 533.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-11-22 15:15:00 | 541.95 | 2023-11-23 09:15:00 | 536.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-11-29 09:45:00 | 532.25 | 2023-11-30 11:15:00 | 537.25 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-11-29 11:15:00 | 533.30 | 2023-11-30 11:15:00 | 537.25 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-11-29 12:00:00 | 533.15 | 2023-11-30 11:15:00 | 537.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-12-11 12:15:00 | 547.90 | 2023-12-13 09:15:00 | 543.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-12 12:15:00 | 548.30 | 2023-12-13 09:15:00 | 543.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-12-12 13:00:00 | 548.00 | 2023-12-13 09:15:00 | 543.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-12-26 12:15:00 | 531.95 | 2023-12-27 14:15:00 | 534.35 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-12-27 10:30:00 | 531.45 | 2023-12-27 14:15:00 | 534.35 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-01-02 14:30:00 | 555.80 | 2024-01-03 12:15:00 | 549.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-01-23 09:45:00 | 535.50 | 2024-01-29 13:15:00 | 535.25 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-01-25 09:15:00 | 535.10 | 2024-01-29 13:15:00 | 535.25 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-01-31 13:15:00 | 538.80 | 2024-02-05 09:15:00 | 540.90 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-01-31 15:15:00 | 540.35 | 2024-02-05 09:15:00 | 540.90 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-02-08 11:00:00 | 534.90 | 2024-02-09 14:15:00 | 539.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-02-14 10:15:00 | 537.90 | 2024-02-22 10:15:00 | 541.15 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-02-14 11:30:00 | 538.50 | 2024-02-22 10:15:00 | 541.15 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-02-27 14:15:00 | 534.50 | 2024-02-28 09:15:00 | 538.25 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-02-28 10:45:00 | 534.00 | 2024-02-28 13:15:00 | 539.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-03-15 10:00:00 | 527.45 | 2024-03-15 12:15:00 | 528.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-03-15 12:15:00 | 527.15 | 2024-03-15 12:15:00 | 528.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-03-20 10:00:00 | 516.95 | 2024-03-21 12:15:00 | 524.85 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-03-20 10:30:00 | 515.40 | 2024-03-21 12:15:00 | 524.85 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-04-12 12:00:00 | 503.80 | 2024-04-16 13:15:00 | 504.35 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-04-16 12:15:00 | 504.00 | 2024-04-16 13:15:00 | 504.35 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-04-16 13:00:00 | 502.95 | 2024-04-16 13:15:00 | 504.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-05-13 15:15:00 | 549.00 | 2024-05-14 09:15:00 | 557.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-05-21 09:15:00 | 536.30 | 2024-05-21 15:15:00 | 541.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-06-03 14:00:00 | 546.75 | 2024-06-04 09:15:00 | 563.60 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-06-03 14:30:00 | 546.10 | 2024-06-04 09:15:00 | 563.60 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-06-20 13:30:00 | 599.70 | 2024-06-26 10:15:00 | 600.50 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-06-21 10:30:00 | 598.60 | 2024-06-26 10:15:00 | 600.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-06-21 11:15:00 | 599.65 | 2024-06-26 12:15:00 | 601.55 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-06-25 10:15:00 | 599.10 | 2024-06-26 12:15:00 | 601.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-06-25 14:45:00 | 593.55 | 2024-06-26 12:15:00 | 601.55 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-06-26 09:15:00 | 592.50 | 2024-06-26 12:15:00 | 601.55 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-07-01 10:30:00 | 610.40 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-01 11:00:00 | 610.60 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-01 11:45:00 | 610.25 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-01 13:15:00 | 609.80 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-03 09:15:00 | 606.80 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-07-05 09:15:00 | 606.50 | 2024-07-05 09:15:00 | 604.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-07-12 10:15:00 | 630.00 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-07-12 10:45:00 | 630.80 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-07-12 11:45:00 | 630.10 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2024-07-12 12:30:00 | 630.20 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-07-15 09:15:00 | 633.25 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-07-16 09:45:00 | 632.20 | 2024-07-19 14:15:00 | 633.15 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-08-09 09:15:00 | 638.90 | 2024-08-09 10:15:00 | 634.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-20 13:15:00 | 622.10 | 2024-08-28 09:15:00 | 639.55 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2024-08-21 09:15:00 | 623.45 | 2024-08-28 09:15:00 | 639.55 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2024-08-30 10:15:00 | 640.60 | 2024-09-03 10:15:00 | 646.65 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-08-30 10:45:00 | 639.65 | 2024-09-03 10:15:00 | 646.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-08-30 13:00:00 | 639.95 | 2024-09-03 10:15:00 | 646.65 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-09-02 11:00:00 | 638.00 | 2024-09-03 10:15:00 | 646.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-09-06 09:15:00 | 647.45 | 2024-09-13 10:15:00 | 657.90 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2024-09-06 14:30:00 | 646.00 | 2024-09-13 10:15:00 | 657.90 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2024-09-09 09:15:00 | 651.30 | 2024-09-13 10:15:00 | 657.90 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-09-18 11:00:00 | 655.85 | 2024-09-19 14:15:00 | 664.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-09-18 12:15:00 | 656.30 | 2024-09-19 14:15:00 | 664.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-09-30 12:45:00 | 629.95 | 2024-10-03 09:15:00 | 598.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:30:00 | 630.00 | 2024-10-03 09:15:00 | 598.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 12:45:00 | 629.95 | 2024-10-07 10:15:00 | 566.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 13:30:00 | 630.00 | 2024-10-07 10:15:00 | 567.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-31 10:15:00 | 540.75 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-10-31 13:30:00 | 540.90 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-10-31 14:45:00 | 539.90 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-11-04 09:15:00 | 539.35 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-11-04 10:15:00 | 534.45 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-11-04 14:00:00 | 534.90 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-11-04 14:30:00 | 534.15 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-05 15:15:00 | 534.90 | 2024-11-06 14:15:00 | 539.35 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-11-07 14:30:00 | 533.50 | 2024-11-14 10:15:00 | 506.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 531.00 | 2024-11-14 12:15:00 | 504.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 14:30:00 | 533.50 | 2024-11-14 14:15:00 | 508.45 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2024-11-08 09:15:00 | 531.00 | 2024-11-14 14:15:00 | 508.45 | STOP_HIT | 0.50 | 4.25% |
| BUY | retest2 | 2024-11-27 13:30:00 | 526.60 | 2024-12-02 12:15:00 | 522.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-11-27 14:15:00 | 526.95 | 2024-12-02 12:15:00 | 522.10 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-11-28 10:45:00 | 528.35 | 2024-12-02 12:15:00 | 522.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-28 12:00:00 | 527.30 | 2024-12-02 12:15:00 | 522.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-29 14:15:00 | 527.85 | 2024-12-02 12:15:00 | 522.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-12-05 14:45:00 | 520.10 | 2024-12-05 15:15:00 | 524.95 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2024-12-19 09:15:00 | 502.50 | 2024-12-20 10:15:00 | 507.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-12-30 14:15:00 | 505.85 | 2025-01-01 09:15:00 | 507.70 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-12-30 15:00:00 | 503.70 | 2025-01-01 09:15:00 | 507.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-31 09:15:00 | 505.40 | 2025-01-01 09:15:00 | 507.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-01-13 12:15:00 | 518.45 | 2025-01-13 12:15:00 | 515.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-01-14 12:15:00 | 515.20 | 2025-01-16 10:15:00 | 517.30 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-01-21 09:15:00 | 524.70 | 2025-01-21 15:15:00 | 520.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-01-21 10:30:00 | 522.40 | 2025-01-21 15:15:00 | 520.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-02-10 11:30:00 | 529.70 | 2025-02-13 10:15:00 | 525.60 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-02-14 11:15:00 | 518.50 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-02-14 12:30:00 | 519.15 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-02-14 13:30:00 | 519.40 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-02-14 14:45:00 | 519.00 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-02-17 11:00:00 | 515.50 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-02-18 10:15:00 | 514.35 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-02-18 11:45:00 | 516.00 | 2025-02-25 11:15:00 | 511.25 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest1 | 2025-03-04 09:15:00 | 487.00 | 2025-03-05 11:15:00 | 490.05 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-03-05 13:30:00 | 487.55 | 2025-03-06 09:15:00 | 492.35 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-03-13 11:00:00 | 501.85 | 2025-03-17 10:15:00 | 492.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-03-13 11:45:00 | 501.50 | 2025-03-17 10:15:00 | 492.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-03-26 13:30:00 | 509.60 | 2025-03-28 14:15:00 | 506.45 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-03-27 10:30:00 | 509.35 | 2025-03-28 14:15:00 | 506.45 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-03-28 11:45:00 | 509.45 | 2025-03-28 14:15:00 | 506.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-04-01 11:15:00 | 501.75 | 2025-04-03 09:15:00 | 476.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:15:00 | 501.75 | 2025-04-07 09:15:00 | 451.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-22 09:15:00 | 477.30 | 2025-04-28 11:15:00 | 483.90 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-05-15 12:15:00 | 468.15 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-15 14:00:00 | 468.55 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-05-16 09:30:00 | 468.05 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-21 09:15:00 | 480.10 | 2025-05-22 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-03 09:15:00 | 485.10 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-06-03 13:00:00 | 485.50 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-04 09:15:00 | 490.30 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-17 11:45:00 | 470.65 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-17 13:30:00 | 470.55 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-18 09:45:00 | 469.75 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-06-18 13:30:00 | 470.25 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-07-02 12:15:00 | 486.50 | 2025-07-11 15:15:00 | 535.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 13:45:00 | 486.00 | 2025-07-11 15:15:00 | 534.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-25 11:30:00 | 510.80 | 2025-07-28 10:15:00 | 520.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-29 14:00:00 | 517.70 | 2025-08-05 09:15:00 | 521.80 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-07-29 15:00:00 | 517.70 | 2025-08-05 09:15:00 | 521.80 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-08-08 12:00:00 | 510.35 | 2025-08-18 09:15:00 | 525.50 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-08-08 13:30:00 | 510.50 | 2025-08-18 09:15:00 | 525.50 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-09-05 13:15:00 | 546.50 | 2025-09-09 09:15:00 | 541.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-08 09:30:00 | 547.75 | 2025-09-09 09:15:00 | 541.25 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-16 11:15:00 | 538.55 | 2025-09-19 13:15:00 | 536.75 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-09-16 11:45:00 | 539.50 | 2025-09-24 10:15:00 | 511.62 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-09-16 12:45:00 | 539.05 | 2025-09-24 10:15:00 | 512.52 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-19 09:30:00 | 537.65 | 2025-09-24 10:15:00 | 512.10 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-09-16 11:45:00 | 539.50 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-09-16 12:45:00 | 539.05 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-09-19 09:30:00 | 537.65 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-09-19 12:00:00 | 533.90 | 2025-09-25 14:15:00 | 510.77 | PARTIAL | 0.50 | 4.33% |
| SELL | retest2 | 2025-09-22 09:15:00 | 532.50 | 2025-09-25 14:15:00 | 505.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:00:00 | 533.90 | 2025-10-01 10:15:00 | 483.88 | TARGET_HIT | 0.50 | 9.37% |
| SELL | retest2 | 2025-09-22 09:15:00 | 532.50 | 2025-10-01 12:15:00 | 490.15 | STOP_HIT | 0.50 | 7.95% |
| BUY | retest2 | 2025-11-12 11:15:00 | 523.55 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-12 12:00:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-12 13:30:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-12 14:45:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-13 11:45:00 | 521.85 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-14 10:30:00 | 522.05 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-05 09:30:00 | 505.70 | 2025-12-05 13:15:00 | 509.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-05 11:15:00 | 507.20 | 2025-12-05 13:15:00 | 509.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-12-09 09:15:00 | 499.50 | 2025-12-10 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-12-09 11:30:00 | 503.50 | 2025-12-10 09:15:00 | 505.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-10 12:00:00 | 499.60 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-10 15:15:00 | 500.00 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-12 10:00:00 | 499.10 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-22 11:15:00 | 493.65 | 2025-12-22 12:15:00 | 496.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-22 12:00:00 | 493.20 | 2025-12-22 12:15:00 | 496.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-23 13:15:00 | 492.70 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-24 10:45:00 | 492.55 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-12-30 13:00:00 | 492.45 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-02 09:15:00 | 501.75 | 2026-01-13 12:15:00 | 518.65 | STOP_HIT | 1.00 | 3.37% |
| SELL | retest2 | 2026-01-19 09:15:00 | 510.35 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-20 09:45:00 | 511.00 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-20 10:45:00 | 511.40 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-29 09:30:00 | 509.50 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-01-30 10:00:00 | 514.70 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2026-01-30 11:15:00 | 514.25 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2026-02-12 12:45:00 | 520.10 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-12 13:30:00 | 520.20 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-19 11:30:00 | 507.70 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-20 11:00:00 | 507.25 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-27 11:45:00 | 517.25 | 2026-03-02 09:15:00 | 510.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2026-03-06 09:15:00 | 485.95 | 2026-03-10 10:15:00 | 478.15 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-13 14:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 09:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-04-01 13:30:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-01 14:30:00 | 413.20 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-01 15:15:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-06 11:15:00 | 413.50 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-04-06 12:30:00 | 409.05 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-04-21 10:15:00 | 453.20 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-21 15:15:00 | 452.55 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-05-04 13:15:00 | 444.60 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-05-04 13:45:00 | 445.00 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.90% |
