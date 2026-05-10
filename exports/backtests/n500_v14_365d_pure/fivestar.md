# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 462.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 21
- **Target hits / Stop hits / Partials:** 0 / 21 / 0
- **Avg / median % per leg:** -3.01% / -2.44%
- **Sum % (uncompounded):** -63.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.49% | -29.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.49% | -29.9% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.69% | -33.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.69% | -33.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 0 | 0.0% | 0 | 21 | 0 | -3.01% | -63.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 677.40 | 716.86 | 716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 675.50 | 715.03 | 715.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 703.60 | 701.97 | 708.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:30:00 | 705.40 | 701.97 | 708.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 708.75 | 702.06 | 708.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 709.35 | 702.06 | 708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 707.70 | 702.11 | 708.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 707.00 | 702.11 | 708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 706.65 | 702.16 | 708.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 701.20 | 702.26 | 708.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 713.40 | 702.37 | 708.33 | SL hit (close>static) qty=1.00 sl=709.45 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 705.00 | 702.84 | 708.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:00:00 | 704.85 | 702.84 | 708.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 702.10 | 702.80 | 708.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 708.60 | 700.65 | 706.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 708.60 | 700.65 | 706.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 708.00 | 700.72 | 706.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 709.60 | 700.72 | 706.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 720.80 | 700.92 | 706.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 720.80 | 700.92 | 706.79 | SL hit (close>static) qty=1.00 sl=709.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 720.80 | 700.92 | 706.79 | SL hit (close>static) qty=1.00 sl=709.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 720.80 | 700.92 | 706.79 | SL hit (close>static) qty=1.00 sl=709.45 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 718.00 | 700.92 | 706.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 718.50 | 701.09 | 706.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:00:00 | 701.00 | 701.59 | 706.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 746.15 | 699.29 | 705.06 | SL hit (close>static) qty=1.00 sl=725.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 798.10 | 710.94 | 710.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 800.05 | 711.83 | 711.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 723.85 | 733.71 | 724.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 715.00 | 733.52 | 724.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 735.90 | 733.52 | 724.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:30:00 | 732.30 | 733.19 | 724.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 724.50 | 750.27 | 740.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:45:00 | 724.85 | 748.66 | 740.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 699.55 | 743.91 | 738.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 699.55 | 743.91 | 738.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 699.55 | 743.91 | 738.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 699.55 | 743.91 | 738.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 650.05 | 733.35 | 733.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 13:15:00 | 645.15 | 732.47 | 732.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 540.30 | 536.65 | 567.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:45:00 | 540.60 | 536.65 | 567.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 584.10 | 536.46 | 561.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 584.10 | 536.46 | 561.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 602.10 | 537.12 | 561.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 600.60 | 537.12 | 561.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 625.95 | 581.08 | 580.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 628.45 | 584.22 | 582.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 596.80 | 597.25 | 590.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 12:30:00 | 598.50 | 597.25 | 590.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 591.25 | 597.12 | 590.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 584.65 | 597.12 | 590.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 591.75 | 600.17 | 592.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 591.75 | 600.17 | 592.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 588.20 | 600.05 | 592.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 588.20 | 600.05 | 592.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 590.00 | 599.95 | 592.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 595.10 | 599.88 | 592.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 592.00 | 599.78 | 592.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 592.40 | 599.78 | 592.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 594.10 | 599.61 | 592.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 589.45 | 599.51 | 592.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 585.50 | 599.37 | 592.83 | SL hit (close<static) qty=1.00 sl=587.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 585.50 | 599.37 | 592.83 | SL hit (close<static) qty=1.00 sl=587.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 585.50 | 599.37 | 592.83 | SL hit (close<static) qty=1.00 sl=587.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 585.50 | 599.37 | 592.83 | SL hit (close<static) qty=1.00 sl=587.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 585.50 | 599.37 | 592.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 590.85 | 599.29 | 592.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 592.40 | 595.76 | 591.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 595.10 | 595.62 | 591.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 592.50 | 595.59 | 591.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 15:00:00 | 591.20 | 595.54 | 591.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 591.00 | 595.50 | 591.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 573.70 | 595.50 | 591.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 580.55 | 595.35 | 591.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 580.55 | 595.35 | 591.54 | SL hit (close<static) qty=1.00 sl=585.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 580.55 | 595.35 | 591.54 | SL hit (close<static) qty=1.00 sl=585.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 580.55 | 595.35 | 591.54 | SL hit (close<static) qty=1.00 sl=585.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 580.55 | 595.35 | 591.54 | SL hit (close<static) qty=1.00 sl=585.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 571.00 | 595.35 | 591.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 567.70 | 588.18 | 588.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 553.50 | 584.68 | 586.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 404.20 | 386.12 | 423.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 09:45:00 | 405.40 | 386.12 | 423.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 424.00 | 389.09 | 422.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 424.00 | 389.09 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 422.55 | 389.42 | 422.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 427.85 | 389.42 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 428.85 | 389.81 | 422.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 412.60 | 392.18 | 422.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 421.10 | 392.67 | 422.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 421.90 | 392.96 | 422.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 420.90 | 393.53 | 422.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 420.65 | 393.80 | 422.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 437.80 | 394.92 | 422.90 | SL hit (close>static) qty=1.00 sl=434.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 437.80 | 394.92 | 422.90 | SL hit (close>static) qty=1.00 sl=434.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 437.80 | 394.92 | 422.90 | SL hit (close>static) qty=1.00 sl=434.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 437.80 | 394.92 | 422.90 | SL hit (close>static) qty=1.00 sl=434.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 495.00 | 442.04 | 441.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 502.00 | 444.70 | 443.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-27 09:15:00 | 701.20 | 2025-05-27 09:15:00 | 713.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-28 09:30:00 | 705.00 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-05-28 10:00:00 | 704.85 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-28 11:30:00 | 702.10 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-06-02 15:00:00 | 701.00 | 2025-06-06 12:15:00 | 746.15 | STOP_HIT | 1.00 | -6.44% |
| BUY | retest2 | 2025-06-23 09:15:00 | 735.90 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2025-06-23 14:30:00 | 732.30 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-07-21 11:00:00 | 724.50 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-22 11:45:00 | 724.85 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-11-27 13:30:00 | 595.10 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-27 15:15:00 | 592.00 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-28 13:00:00 | 592.40 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-28 14:30:00 | 594.10 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-04 10:30:00 | 592.40 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-04 13:15:00 | 595.10 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-04 14:00:00 | 592.50 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-12-04 15:00:00 | 591.20 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-13 09:15:00 | 412.60 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2026-04-13 11:00:00 | 421.10 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-04-13 12:15:00 | 421.90 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-04-13 14:15:00 | 420.90 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -4.02% |
