# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 515.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 1 |
| TARGET_HIT | 7 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 27
- **Target hits / Stop hits / Partials:** 7 / 29 / 1
- **Avg / median % per leg:** 0.67% / -1.16%
- **Sum % (uncompounded):** 24.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 7 | 100.0% | 7 | 0 | 0 | 10.00% | 70.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 7 | 100.0% | 7 | 0 | 0 | 10.00% | 70.0% |
| SELL (all) | 30 | 3 | 10.0% | 0 | 29 | 1 | -1.50% | -45.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 3 | 10.0% | 0 | 29 | 1 | -1.50% | -45.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 10 | 27.0% | 7 | 29 | 1 | 0.67% | 24.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 526.30 | 512.38 | 512.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 529.75 | 512.92 | 512.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 10:15:00 | 526.50 | 529.23 | 522.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 11:00:00 | 526.50 | 529.23 | 522.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 522.65 | 529.12 | 522.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 522.65 | 529.12 | 522.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 521.35 | 529.05 | 522.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 521.40 | 529.05 | 522.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 518.60 | 528.94 | 522.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 518.60 | 528.94 | 522.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 522.40 | 528.70 | 522.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:30:00 | 521.60 | 528.70 | 522.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 520.10 | 528.62 | 522.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 520.10 | 528.62 | 522.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 516.65 | 528.50 | 522.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:45:00 | 515.85 | 528.50 | 522.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 534.45 | 528.50 | 522.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 522.55 | 528.50 | 522.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 525.50 | 528.52 | 522.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 523.50 | 528.52 | 522.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 530.95 | 528.52 | 522.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 534.70 | 528.59 | 522.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:45:00 | 535.50 | 529.09 | 523.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-22 09:15:00 | 588.17 | 538.44 | 529.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 533.30 | 569.62 | 569.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 523.75 | 563.73 | 566.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 468.80 | 464.91 | 487.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 468.80 | 464.91 | 487.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 479.65 | 466.05 | 481.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 479.65 | 466.05 | 481.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 482.75 | 466.21 | 481.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 483.65 | 466.21 | 481.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 481.40 | 466.36 | 481.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 484.15 | 466.36 | 481.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 478.90 | 466.49 | 481.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:00:00 | 476.95 | 466.59 | 481.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 482.25 | 466.75 | 481.20 | SL hit (close>static) qty=1.00 sl=481.55 alert=retest2 |

### Cycle 3 — BUY (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 09:15:00 | 490.40 | 483.48 | 483.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 506.75 | 485.01 | 484.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 485.30 | 488.49 | 486.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 485.30 | 488.49 | 486.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 485.30 | 488.49 | 486.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 485.30 | 488.49 | 486.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 483.50 | 488.44 | 486.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 483.50 | 488.44 | 486.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 482.45 | 488.38 | 486.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 482.45 | 488.38 | 486.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 487.70 | 488.10 | 486.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 486.95 | 488.10 | 486.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 489.15 | 488.11 | 486.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:30:00 | 487.30 | 488.11 | 486.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 485.00 | 488.09 | 486.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 493.65 | 488.09 | 486.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 497.40 | 488.18 | 486.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 500.15 | 488.30 | 486.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 10:00:00 | 498.50 | 489.08 | 486.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 499.65 | 492.29 | 488.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 499.55 | 495.23 | 490.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 492.20 | 495.97 | 491.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 491.15 | 495.97 | 491.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 493.75 | 495.94 | 491.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 495.50 | 495.91 | 491.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-09 09:15:00 | 550.16 | 501.36 | 495.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 536.50 | 560.20 | 560.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 530.65 | 556.78 | 558.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 544.70 | 543.98 | 549.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:45:00 | 544.65 | 543.98 | 549.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 550.55 | 544.04 | 549.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 550.55 | 544.04 | 549.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 552.00 | 544.12 | 549.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 550.35 | 544.12 | 549.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 548.25 | 544.23 | 549.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 546.80 | 544.23 | 549.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 551.40 | 544.32 | 549.72 | SL hit (close>static) qty=1.00 sl=550.45 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 581.00 | 541.82 | 541.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 587.00 | 547.28 | 544.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 560.30 | 560.92 | 553.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 560.30 | 560.92 | 553.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 553.20 | 560.67 | 553.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 554.40 | 560.67 | 553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 552.05 | 560.58 | 553.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 551.10 | 560.58 | 553.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 551.20 | 560.49 | 553.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 551.20 | 560.49 | 553.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 553.50 | 560.32 | 553.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 551.40 | 560.32 | 553.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 558.50 | 560.31 | 553.47 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 537.80 | 549.63 | 549.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 548.00 | 548.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 431.00 | 430.76 | 454.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 431.00 | 430.76 | 454.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 452.15 | 431.78 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 452.15 | 431.78 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 452.00 | 431.98 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 452.00 | 431.98 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 454.50 | 432.20 | 452.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 454.50 | 432.20 | 452.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 452.80 | 432.41 | 452.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 445.70 | 433.03 | 452.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 455.15 | 433.40 | 452.23 | SL hit (close>static) qty=1.00 sl=454.70 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 517.60 | 461.55 | 461.51 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-13 10:30:00 | 534.70 | 2024-08-22 09:15:00 | 588.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 11:45:00 | 535.50 | 2024-08-22 09:15:00 | 589.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 476.95 | 2025-01-23 14:15:00 | 482.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-27 09:15:00 | 476.50 | 2025-01-27 10:15:00 | 482.95 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-01-28 09:15:00 | 477.40 | 2025-02-01 12:15:00 | 490.85 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-01-28 10:00:00 | 475.60 | 2025-02-01 12:15:00 | 490.85 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-02-05 09:15:00 | 473.65 | 2025-02-17 12:15:00 | 481.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-10 15:00:00 | 484.05 | 2025-02-17 12:15:00 | 481.90 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-02-11 09:15:00 | 480.55 | 2025-02-24 12:15:00 | 497.35 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-02-12 14:00:00 | 484.70 | 2025-02-24 12:15:00 | 497.35 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-02-14 12:15:00 | 478.00 | 2025-02-24 12:15:00 | 497.35 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-02-14 15:00:00 | 478.20 | 2025-02-24 12:15:00 | 497.35 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-03-17 11:00:00 | 500.15 | 2025-04-09 09:15:00 | 550.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 10:00:00 | 498.50 | 2025-04-09 09:15:00 | 548.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 14:00:00 | 499.65 | 2025-04-09 09:15:00 | 549.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-27 14:45:00 | 499.55 | 2025-04-09 09:15:00 | 549.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-02 10:15:00 | 495.50 | 2025-04-09 09:15:00 | 545.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-12 11:15:00 | 546.80 | 2025-09-12 12:15:00 | 551.40 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-12 12:30:00 | 547.30 | 2025-09-16 09:15:00 | 551.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-12 14:15:00 | 546.65 | 2025-09-16 09:15:00 | 551.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-16 12:45:00 | 546.80 | 2025-09-26 15:15:00 | 519.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:45:00 | 546.80 | 2025-10-03 12:15:00 | 539.15 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-10-06 09:15:00 | 535.20 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 09:45:00 | 536.70 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-17 12:45:00 | 536.65 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-10-17 13:15:00 | 536.95 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-20 10:45:00 | 541.40 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-20 11:45:00 | 541.50 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-20 12:30:00 | 541.35 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-20 13:15:00 | 541.40 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-03 11:30:00 | 536.00 | 2025-11-06 09:15:00 | 550.80 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-11-07 10:00:00 | 535.80 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-11-10 12:30:00 | 536.00 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-10 13:00:00 | 535.00 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-11-11 11:45:00 | 544.55 | 2025-11-12 09:15:00 | 551.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-11 15:15:00 | 545.00 | 2025-11-12 09:15:00 | 551.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-13 09:15:00 | 445.70 | 2026-04-13 10:15:00 | 455.15 | STOP_HIT | 1.00 | -2.12% |
