# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 715.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 6 |
| TARGET_HIT | 10 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 18
- **Target hits / Stop hits / Partials:** 10 / 18 / 6
- **Avg / median % per leg:** 2.65% / -0.62%
- **Sum % (uncompounded):** 90.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 30 | 12 | 40.0% | 6 | 18 | 6 | 1.67% | 50.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 12 | 40.0% | 6 | 18 | 6 | 1.67% | 50.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 16 | 47.1% | 10 | 18 | 6 | 2.65% | 90.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 12:15:00 | 374.00 | 383.29 | 383.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 13:15:00 | 371.80 | 383.18 | 383.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 379.45 | 379.29 | 381.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 379.45 | 379.29 | 381.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 379.45 | 379.29 | 381.01 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 390.98 | 382.39 | 382.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 394.00 | 383.29 | 382.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 382.76 | 386.24 | 384.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 13:15:00 | 382.76 | 386.24 | 384.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 382.76 | 386.24 | 384.52 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 408.33 | 415.10 | 415.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 402.01 | 414.28 | 414.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 12:15:00 | 405.63 | 405.44 | 409.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-24 12:45:00 | 405.64 | 405.44 | 409.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 407.33 | 405.46 | 409.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 15:00:00 | 404.09 | 405.55 | 409.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 09:45:00 | 402.72 | 405.48 | 409.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 15:15:00 | 404.00 | 405.00 | 408.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 09:45:00 | 403.34 | 404.98 | 408.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 403.51 | 404.94 | 408.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:45:00 | 403.63 | 404.94 | 408.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 409.42 | 404.32 | 407.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 409.42 | 404.32 | 407.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 416.56 | 404.44 | 408.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 416.56 | 404.44 | 408.03 | SL hit (close>static) qty=1.00 sl=409.60 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 427.79 | 397.37 | 397.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 429.69 | 399.46 | 398.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 422.35 | 422.47 | 414.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 422.35 | 422.47 | 414.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 422.35 | 422.47 | 414.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 428.49 | 422.07 | 415.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 15:15:00 | 424.60 | 422.37 | 415.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 12:45:00 | 424.55 | 422.43 | 415.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-16 11:15:00 | 467.06 | 430.10 | 421.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 543.35 | 596.03 | 596.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 540.15 | 594.95 | 595.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 552.00 | 551.20 | 568.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 542.30 | 551.20 | 568.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 570.00 | 550.88 | 568.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 570.00 | 550.88 | 568.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 575.05 | 551.12 | 568.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 567.70 | 551.12 | 568.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 580.25 | 551.55 | 568.12 | SL hit (close>static) qty=1.00 sl=579.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 620.00 | 579.68 | 579.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 625.25 | 580.52 | 579.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 586.65 | 588.68 | 584.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 14:00:00 | 586.65 | 588.68 | 584.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 593.10 | 588.72 | 584.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 604.65 | 587.10 | 584.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 12:15:00 | 665.12 | 600.95 | 591.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 701.45 | 722.68 | 722.75 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 727.35 | 722.71 | 722.69 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 718.25 | 722.67 | 722.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 716.60 | 722.61 | 722.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 725.00 | 722.62 | 722.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 724.20 | 722.64 | 722.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 722.25 | 722.61 | 722.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 719.50 | 722.61 | 722.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 728.65 | 722.40 | 722.54 | SL hit (close>static) qty=1.00 sl=725.60 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 732.85 | 722.73 | 722.69 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 708.80 | 722.64 | 722.68 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 727.65 | 722.75 | 722.74 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 722.59 | 722.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 722.35 | 722.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 702.00 | 691.75 | 703.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 700.00 | 691.93 | 702.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 693.90 | 691.93 | 702.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 697.05 | 692.01 | 702.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 708.95 | 692.73 | 702.50 | SL hit (close>static) qty=1.00 sl=708.00 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 699.95 | 650.73 | 650.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 710.70 | 653.05 | 651.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 683.75 | 684.66 | 671.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 683.75 | 684.66 | 671.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 669.00 | 684.34 | 671.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 667.80 | 684.34 | 671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 662.00 | 684.12 | 671.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:45:00 | 664.00 | 684.12 | 671.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 670.15 | 675.93 | 668.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 670.65 | 675.93 | 668.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 660.50 | 675.61 | 668.97 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 634.40 | 663.34 | 663.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 619.90 | 662.36 | 662.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 669.35 | 651.98 | 657.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 663.20 | 652.09 | 657.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 661.00 | 652.68 | 657.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 661.85 | 653.22 | 657.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 661.55 | 653.30 | 657.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:00:00 | 661.95 | 653.46 | 657.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 660.40 | 654.19 | 657.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 660.40 | 654.19 | 657.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 658.65 | 654.23 | 657.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 657.00 | 654.31 | 657.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 654.00 | 654.35 | 657.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 673.70 | 654.54 | 657.87 | SL hit (close>static) qty=1.00 sl=673.00 alert=retest2 |

### Cycle 16 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 680.85 | 660.77 | 660.68 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 646.55 | 660.55 | 660.60 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 674.65 | 660.73 | 660.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 683.90 | 662.67 | 661.74 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-25 15:00:00 | 404.09 | 2024-05-06 11:15:00 | 416.56 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-04-26 09:45:00 | 402.72 | 2024-05-06 11:15:00 | 416.56 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-04-29 15:15:00 | 404.00 | 2024-05-06 11:15:00 | 416.56 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-04-30 09:45:00 | 403.34 | 2024-05-06 11:15:00 | 416.56 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-05-09 13:00:00 | 403.07 | 2024-05-17 09:15:00 | 382.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 13:45:00 | 403.01 | 2024-05-17 09:15:00 | 382.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-10 10:00:00 | 401.72 | 2024-05-17 09:15:00 | 381.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-10 12:00:00 | 401.71 | 2024-05-17 09:15:00 | 381.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 13:00:00 | 403.07 | 2024-05-27 11:15:00 | 362.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-09 13:45:00 | 403.01 | 2024-05-27 11:15:00 | 362.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-10 10:00:00 | 401.72 | 2024-05-27 11:15:00 | 361.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-10 12:00:00 | 401.71 | 2024-05-27 11:15:00 | 361.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 428.49 | 2024-08-16 11:15:00 | 467.06 | TARGET_HIT | 1.00 | 9.00% |
| BUY | retest2 | 2024-08-06 15:15:00 | 424.60 | 2024-08-16 11:15:00 | 467.01 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2024-08-07 12:45:00 | 424.55 | 2024-08-19 09:15:00 | 471.34 | TARGET_HIT | 1.00 | 11.02% |
| SELL | retest2 | 2025-03-17 09:15:00 | 567.70 | 2025-03-17 10:15:00 | 580.25 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-04-11 09:15:00 | 604.65 | 2025-04-21 12:15:00 | 665.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 14:30:00 | 722.25 | 2025-10-28 09:15:00 | 728.65 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-27 10:15:00 | 719.50 | 2025-10-28 09:15:00 | 728.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-28 11:45:00 | 721.00 | 2025-10-29 12:15:00 | 726.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-28 14:30:00 | 721.70 | 2025-10-29 12:15:00 | 726.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-03 09:15:00 | 693.90 | 2025-12-04 10:15:00 | 708.95 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-12-03 12:15:00 | 697.05 | 2025-12-04 10:15:00 | 708.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-08 10:00:00 | 696.95 | 2025-12-10 14:15:00 | 662.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:45:00 | 696.55 | 2025-12-10 14:15:00 | 661.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:00:00 | 696.95 | 2025-12-24 15:15:00 | 627.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-08 10:45:00 | 696.55 | 2025-12-26 10:15:00 | 626.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-09 14:00:00 | 636.55 | 2026-02-10 09:15:00 | 665.45 | STOP_HIT | 1.00 | -4.54% |
| SELL | retest2 | 2026-04-08 15:15:00 | 661.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-09 13:30:00 | 661.85 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-04-09 15:00:00 | 661.55 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-10 10:00:00 | 661.95 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-13 14:00:00 | 657.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-04-13 15:15:00 | 654.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -3.01% |
