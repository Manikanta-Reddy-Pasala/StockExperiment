# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 537.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 63 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 19 / 51
- **Target hits / Stop hits / Partials:** 2 / 57 / 11
- **Avg / median % per leg:** -0.89% / -1.69%
- **Sum % (uncompounded):** -62.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -2.45% | -49.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 0 | 0.0% | 0 | 20 | 0 | -2.45% | -49.0% |
| SELL (all) | 50 | 19 | 38.0% | 2 | 37 | 11 | -0.27% | -13.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 19 | 38.0% | 2 | 37 | 11 | -0.27% | -13.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 19 | 27.1% | 2 | 57 | 11 | -0.89% | -62.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 506.00 | 538.40 | 538.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 499.85 | 535.84 | 537.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 532.95 | 532.46 | 535.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 12:00:00 | 532.95 | 532.46 | 535.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 534.60 | 532.48 | 535.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:15:00 | 531.00 | 532.48 | 535.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 504.45 | 530.28 | 533.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 15:15:00 | 477.90 | 523.48 | 530.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 565.35 | 517.90 | 517.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 571.50 | 518.91 | 518.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 539.95 | 542.35 | 531.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-27 11:00:00 | 539.95 | 542.35 | 531.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 521.95 | 542.24 | 532.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:45:00 | 522.05 | 542.24 | 532.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 522.25 | 542.04 | 532.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:15:00 | 520.55 | 542.04 | 532.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 539.00 | 542.01 | 532.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:30:00 | 519.95 | 542.01 | 532.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 537.60 | 541.89 | 532.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:30:00 | 533.65 | 541.89 | 532.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 532.60 | 541.88 | 532.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 532.60 | 541.88 | 532.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 545.95 | 541.92 | 532.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:30:00 | 554.30 | 542.04 | 532.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:30:00 | 565.65 | 542.04 | 533.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 12:15:00 | 525.95 | 552.03 | 541.16 | SL hit (close<static) qty=1.00 sl=530.95 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 485.00 | 534.63 | 534.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 479.40 | 533.20 | 533.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 528.00 | 524.71 | 529.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:00:00 | 528.00 | 524.71 | 529.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 531.50 | 524.61 | 528.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 533.35 | 524.61 | 528.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 521.80 | 524.58 | 528.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 14:15:00 | 512.50 | 524.48 | 528.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 520.25 | 521.84 | 526.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 520.00 | 521.77 | 526.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 533.95 | 522.16 | 526.72 | SL hit (close>static) qty=1.00 sl=531.70 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 549.80 | 514.52 | 514.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 560.40 | 515.69 | 514.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 529.00 | 530.47 | 523.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 11:45:00 | 528.75 | 530.47 | 523.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 519.25 | 530.91 | 524.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 520.55 | 530.91 | 524.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 520.45 | 530.80 | 524.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 525.00 | 523.33 | 521.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 521.80 | 523.30 | 521.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 512.25 | 523.17 | 521.36 | SL hit (close<static) qty=1.00 sl=517.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 516.00 | 520.26 | 520.26 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 533.50 | 520.31 | 520.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 539.00 | 520.50 | 520.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 520.95 | 521.90 | 521.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 520.95 | 521.90 | 521.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 520.95 | 521.90 | 521.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 520.95 | 521.90 | 521.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 519.50 | 521.87 | 521.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 519.85 | 521.87 | 521.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 519.90 | 521.85 | 521.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 519.90 | 521.85 | 521.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 519.15 | 521.81 | 521.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 519.15 | 521.81 | 521.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 517.00 | 521.76 | 521.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 516.00 | 521.76 | 521.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 518.20 | 521.40 | 520.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:15:00 | 517.00 | 521.40 | 520.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 515.70 | 521.34 | 520.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 515.70 | 521.34 | 520.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 508.80 | 520.37 | 520.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 504.60 | 519.30 | 519.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 508.55 | 496.99 | 506.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 508.55 | 496.99 | 506.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 508.55 | 496.99 | 506.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:00:00 | 504.50 | 499.94 | 506.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 514.40 | 500.60 | 506.83 | SL hit (close>static) qty=1.00 sl=512.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 529.35 | 509.16 | 509.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 540.30 | 510.33 | 509.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 563.15 | 563.38 | 548.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 563.15 | 563.38 | 548.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 565.30 | 585.97 | 573.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 560.85 | 585.97 | 573.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 566.10 | 585.77 | 573.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 565.80 | 585.77 | 573.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 581.95 | 587.84 | 577.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 590.00 | 587.81 | 577.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 572.60 | 589.22 | 580.24 | SL hit (close<static) qty=1.00 sl=577.35 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 516.55 | 577.63 | 577.67 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 11:15:00 | 593.20 | 573.74 | 573.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 599.30 | 574.70 | 574.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 573.00 | 578.32 | 576.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 573.00 | 578.32 | 576.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 573.00 | 578.32 | 576.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 573.00 | 578.32 | 576.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 574.00 | 578.28 | 576.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 567.90 | 578.28 | 576.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 579.70 | 578.20 | 576.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:15:00 | 581.40 | 578.20 | 576.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 584.35 | 578.27 | 576.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 584.60 | 578.25 | 576.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 581.05 | 578.27 | 576.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 574.45 | 578.24 | 576.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 574.45 | 578.24 | 576.35 | SL hit (close<static) qty=1.00 sl=575.80 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 536.05 | 574.76 | 574.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 531.40 | 573.33 | 574.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 15:15:00 | 525.85 | 524.95 | 543.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 547.25 | 525.04 | 543.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 547.25 | 525.04 | 543.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 547.25 | 525.04 | 543.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 542.00 | 525.21 | 543.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 520.60 | 525.21 | 543.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 521.10 | 524.77 | 542.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 494.57 | 522.25 | 539.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 495.05 | 522.25 | 539.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 525.70 | 518.88 | 536.28 | SL hit (close>ema200) qty=0.50 sl=518.88 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-06 14:15:00 | 531.00 | 2024-11-11 09:15:00 | 504.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:15:00 | 531.00 | 2024-11-13 15:15:00 | 477.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 14:30:00 | 525.65 | 2024-12-10 14:15:00 | 542.70 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-12-10 10:30:00 | 529.40 | 2024-12-10 14:15:00 | 542.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-12-10 12:30:00 | 531.40 | 2024-12-10 14:15:00 | 542.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-12-17 14:45:00 | 515.55 | 2024-12-30 09:15:00 | 489.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:45:00 | 515.50 | 2024-12-30 09:15:00 | 489.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 10:45:00 | 515.10 | 2024-12-30 09:15:00 | 489.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 14:00:00 | 515.45 | 2024-12-30 09:15:00 | 489.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:15:00 | 511.15 | 2024-12-30 09:15:00 | 485.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:00:00 | 511.45 | 2024-12-30 09:15:00 | 485.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:30:00 | 509.40 | 2024-12-30 10:15:00 | 483.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 14:45:00 | 515.55 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest2 | 2024-12-18 09:45:00 | 515.50 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.87% |
| SELL | retest2 | 2024-12-18 10:45:00 | 515.10 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2024-12-18 14:00:00 | 515.45 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.86% |
| SELL | retest2 | 2024-12-20 10:15:00 | 511.15 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2024-12-20 13:00:00 | 511.45 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | 0.09% |
| SELL | retest2 | 2024-12-20 14:30:00 | 509.40 | 2025-01-01 11:15:00 | 511.00 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-01-02 11:30:00 | 511.45 | 2025-01-07 09:15:00 | 540.05 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2025-01-03 12:15:00 | 513.10 | 2025-01-07 09:15:00 | 540.05 | STOP_HIT | 1.00 | -5.25% |
| SELL | retest2 | 2025-01-13 14:30:00 | 508.05 | 2025-01-14 09:15:00 | 529.30 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-01-30 09:30:00 | 554.30 | 2025-02-11 12:15:00 | 525.95 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2025-01-31 11:30:00 | 565.65 | 2025-02-11 12:15:00 | 525.95 | STOP_HIT | 1.00 | -7.02% |
| BUY | retest2 | 2025-02-12 13:00:00 | 548.55 | 2025-02-13 14:15:00 | 530.80 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-02-19 11:00:00 | 559.00 | 2025-02-21 09:15:00 | 529.00 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2025-03-10 14:15:00 | 512.50 | 2025-03-19 09:15:00 | 533.95 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-03-17 12:15:00 | 520.25 | 2025-03-19 09:15:00 | 533.95 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-03-17 13:30:00 | 520.00 | 2025-03-19 09:15:00 | 533.95 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-03-20 11:45:00 | 520.15 | 2025-03-20 15:15:00 | 535.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-03-25 12:30:00 | 528.00 | 2025-03-26 09:15:00 | 547.90 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-03-27 12:00:00 | 527.70 | 2025-04-01 14:15:00 | 544.95 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-04-04 09:15:00 | 526.70 | 2025-04-07 09:15:00 | 500.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 526.70 | 2025-04-09 09:15:00 | 474.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-29 09:45:00 | 526.10 | 2025-06-05 09:15:00 | 536.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-06-03 12:15:00 | 511.95 | 2025-06-05 09:15:00 | 536.00 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-06-03 15:00:00 | 511.95 | 2025-06-05 09:15:00 | 536.00 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-06-04 09:30:00 | 512.10 | 2025-06-06 09:15:00 | 550.00 | STOP_HIT | 1.00 | -7.40% |
| BUY | retest2 | 2025-06-30 09:15:00 | 525.00 | 2025-07-01 09:15:00 | 512.25 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-06-30 15:00:00 | 521.80 | 2025-07-01 09:15:00 | 512.25 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-07-01 11:15:00 | 521.50 | 2025-07-01 11:15:00 | 516.70 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-01 13:00:00 | 521.15 | 2025-07-02 14:15:00 | 519.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-02 09:15:00 | 524.95 | 2025-07-02 14:15:00 | 519.05 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-02 10:00:00 | 524.80 | 2025-07-02 14:15:00 | 519.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-02 11:15:00 | 524.90 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-07 14:00:00 | 526.50 | 2025-07-08 09:15:00 | 517.25 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-21 12:00:00 | 504.50 | 2025-08-25 09:15:00 | 514.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-08-26 09:30:00 | 504.50 | 2025-09-02 10:15:00 | 512.80 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-26 14:15:00 | 504.45 | 2025-09-02 10:15:00 | 512.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-28 11:45:00 | 503.00 | 2025-09-02 10:15:00 | 512.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-04 15:00:00 | 504.35 | 2025-09-08 11:15:00 | 524.05 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-09-05 09:15:00 | 503.65 | 2025-09-08 11:15:00 | 524.05 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-09-05 10:45:00 | 503.65 | 2025-09-08 11:15:00 | 524.05 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-09-05 15:00:00 | 503.90 | 2025-09-08 11:15:00 | 524.05 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-09-10 12:30:00 | 510.60 | 2025-09-16 11:15:00 | 516.95 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-10 13:45:00 | 510.15 | 2025-09-16 11:15:00 | 516.95 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-10 15:00:00 | 509.75 | 2025-09-16 11:15:00 | 516.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-12 11:45:00 | 509.55 | 2025-09-16 11:15:00 | 516.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-19 09:15:00 | 590.00 | 2025-12-29 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-02 09:15:00 | 588.55 | 2026-01-12 09:15:00 | 571.15 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-01-09 09:45:00 | 591.50 | 2026-01-12 09:15:00 | 571.15 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2026-02-20 12:15:00 | 581.40 | 2026-02-23 11:15:00 | 574.45 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-20 15:00:00 | 584.35 | 2026-02-23 11:15:00 | 574.45 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-23 09:15:00 | 584.60 | 2026-02-23 11:15:00 | 574.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-23 11:15:00 | 581.05 | 2026-02-23 11:15:00 | 574.45 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-25 15:15:00 | 583.00 | 2026-02-26 12:15:00 | 572.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-27 09:15:00 | 520.60 | 2026-04-02 09:15:00 | 494.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 15:00:00 | 521.10 | 2026-04-02 09:15:00 | 495.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 520.60 | 2026-04-08 09:15:00 | 525.70 | STOP_HIT | 0.50 | -0.98% |
| SELL | retest2 | 2026-03-27 15:00:00 | 521.10 | 2026-04-08 09:15:00 | 525.70 | STOP_HIT | 0.50 | -0.88% |
