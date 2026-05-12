# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 830.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 79 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 66 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 58
- **Target hits / Stop hits / Partials:** 1 / 66 / 4
- **Avg / median % per leg:** -0.92% / -1.44%
- **Sum % (uncompounded):** -65.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 7 | 15.9% | 1 | 42 | 1 | -1.05% | -46.1% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.19% | 6.4% |
| BUY @ 3rd Alert (retest2) | 42 | 5 | 11.9% | 1 | 41 | 0 | -1.25% | -52.4% |
| SELL (all) | 27 | 6 | 22.2% | 0 | 24 | 3 | -0.72% | -19.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 6 | 22.2% | 0 | 24 | 3 | -0.72% | -19.3% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.19% | 6.4% |
| retest2 (combined) | 69 | 11 | 15.9% | 1 | 65 | 3 | -1.04% | -71.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:15:00 | 637.45 | 610.27 | 591.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 12:15:00 | 669.32 | 623.99 | 602.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 646.20 | 653.74 | 629.26 | SL hit (close<ema200) qty=0.50 sl=653.74 alert=retest1 |

### Cycle 2 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 640.00 | 665.57 | 665.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 633.25 | 664.44 | 665.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 631.50 | 628.88 | 642.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 11:00:00 | 631.50 | 628.88 | 642.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 645.85 | 629.18 | 642.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:00:00 | 645.85 | 629.18 | 642.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 646.50 | 629.35 | 642.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:00:00 | 646.50 | 629.35 | 642.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 640.85 | 631.34 | 643.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 643.25 | 631.34 | 643.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 647.75 | 631.50 | 643.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 647.75 | 631.50 | 643.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 647.50 | 631.66 | 643.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:30:00 | 648.80 | 631.66 | 643.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 642.80 | 632.31 | 643.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 643.50 | 632.31 | 643.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 642.55 | 632.41 | 643.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 643.80 | 632.41 | 643.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 642.80 | 632.52 | 643.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 13:00:00 | 639.15 | 632.58 | 643.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 14:15:00 | 645.05 | 632.77 | 643.37 | SL hit (close>static) qty=1.00 sl=644.45 alert=retest2 |

### Cycle 3 — BUY (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 14:15:00 | 674.10 | 641.98 | 641.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 677.40 | 655.57 | 650.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 658.60 | 662.21 | 654.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 658.60 | 662.21 | 654.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 650.55 | 662.07 | 655.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 650.55 | 662.07 | 655.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 652.30 | 661.97 | 654.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 654.20 | 661.85 | 654.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 645.95 | 661.55 | 654.88 | SL hit (close<static) qty=1.00 sl=646.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 626.45 | 649.87 | 649.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 623.10 | 647.47 | 648.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 620.75 | 620.53 | 631.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 14:00:00 | 620.75 | 620.53 | 631.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 630.55 | 620.64 | 630.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:45:00 | 630.00 | 620.64 | 630.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 630.85 | 620.74 | 630.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 10:00:00 | 628.25 | 621.13 | 630.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 629.05 | 621.22 | 630.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 15:15:00 | 632.50 | 621.69 | 630.65 | SL hit (close>static) qty=1.00 sl=631.75 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 13:15:00 | 662.65 | 636.42 | 636.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 669.30 | 637.26 | 636.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 704.45 | 706.03 | 685.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 704.45 | 706.03 | 685.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 693.35 | 705.81 | 694.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 693.35 | 705.81 | 694.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 693.05 | 705.68 | 694.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 693.00 | 705.68 | 694.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 693.10 | 705.56 | 694.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 692.40 | 705.56 | 694.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 692.95 | 702.74 | 694.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 692.95 | 702.74 | 694.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 692.90 | 702.64 | 694.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:30:00 | 692.00 | 702.64 | 694.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 693.75 | 702.55 | 694.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 694.35 | 702.55 | 694.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 692.10 | 702.45 | 694.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 690.80 | 702.45 | 694.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 690.30 | 701.92 | 693.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 689.00 | 701.92 | 693.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 695.35 | 701.19 | 693.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 693.65 | 701.19 | 693.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 690.75 | 701.09 | 693.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 690.75 | 701.09 | 693.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 689.40 | 700.97 | 693.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 689.40 | 700.97 | 693.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 691.65 | 700.49 | 693.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 691.65 | 700.49 | 693.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 689.40 | 700.37 | 693.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 688.45 | 700.37 | 693.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 689.00 | 699.99 | 693.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 690.75 | 699.99 | 693.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 691.00 | 699.90 | 693.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 691.00 | 699.90 | 693.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 693.05 | 699.73 | 693.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 693.05 | 699.73 | 693.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 692.95 | 699.66 | 693.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 700.00 | 699.66 | 693.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 698.75 | 716.66 | 709.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:15:00 | 695.35 | 715.90 | 709.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 695.20 | 714.77 | 708.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 710.75 | 713.35 | 708.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 712.20 | 713.35 | 708.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 712.10 | 713.28 | 708.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 721.50 | 713.23 | 708.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:00:00 | 712.60 | 713.29 | 708.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 710.80 | 713.40 | 708.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 710.00 | 713.40 | 708.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 715.30 | 713.42 | 708.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 708.55 | 713.42 | 708.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 714.20 | 713.58 | 709.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 718.75 | 713.56 | 709.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 706.00 | 713.39 | 709.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 693.70 | 716.49 | 716.58 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 731.10 | 716.12 | 716.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 735.10 | 716.99 | 716.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 718.00 | 718.49 | 717.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 720.00 | 718.50 | 717.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 716.35 | 718.50 | 717.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 719.70 | 718.72 | 717.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 719.70 | 718.72 | 717.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 719.75 | 719.15 | 717.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 725.40 | 718.11 | 717.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 728.00 | 718.33 | 717.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:00:00 | 727.80 | 718.43 | 717.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:00:00 | 725.00 | 728.89 | 724.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 721.35 | 728.82 | 724.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 721.35 | 728.82 | 724.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 724.20 | 728.77 | 724.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 727.95 | 728.76 | 724.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 718.00 | 728.41 | 724.14 | SL hit (close<static) qty=1.00 sl=720.55 alert=retest2 |

### Cycle 8 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 753.15 | 755.64 | 755.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 11:15:00 | 748.00 | 755.52 | 755.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 756.35 | 754.93 | 755.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 755.90 | 754.94 | 755.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 753.10 | 755.41 | 755.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 748.25 | 755.35 | 755.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 759.20 | 754.88 | 755.21 | SL hit (close>static) qty=1.00 sl=758.00 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 762.80 | 755.53 | 755.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 767.60 | 755.73 | 755.63 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 09:15:00 | 637.45 | 2024-07-16 12:15:00 | 669.32 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-08 09:15:00 | 637.45 | 2024-08-06 09:15:00 | 646.20 | STOP_HIT | 0.50 | 1.37% |
| BUY | retest2 | 2024-08-07 09:15:00 | 639.15 | 2024-09-17 09:15:00 | 703.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-28 10:15:00 | 630.30 | 2024-10-29 13:15:00 | 625.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-10-29 14:45:00 | 630.65 | 2024-10-29 15:15:00 | 625.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-30 09:15:00 | 678.00 | 2024-10-31 13:15:00 | 640.00 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2024-12-02 13:00:00 | 639.15 | 2024-12-02 14:15:00 | 645.05 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-12-03 09:15:00 | 640.80 | 2024-12-09 09:15:00 | 608.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 10:30:00 | 641.00 | 2024-12-09 09:15:00 | 608.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 12:15:00 | 640.05 | 2024-12-09 09:15:00 | 608.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 09:15:00 | 640.80 | 2024-12-11 11:15:00 | 631.80 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2024-12-03 10:30:00 | 641.00 | 2024-12-11 11:15:00 | 631.80 | STOP_HIT | 0.50 | 1.44% |
| SELL | retest2 | 2024-12-03 12:15:00 | 640.05 | 2024-12-11 11:15:00 | 631.80 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2024-12-12 13:00:00 | 632.00 | 2024-12-19 15:15:00 | 642.75 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-13 10:30:00 | 632.25 | 2024-12-19 15:15:00 | 642.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-17 15:15:00 | 630.00 | 2025-01-01 12:15:00 | 643.95 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-12-18 09:45:00 | 632.10 | 2025-01-01 12:15:00 | 643.95 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-12-19 09:15:00 | 634.00 | 2025-01-01 13:15:00 | 646.05 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-12-19 09:45:00 | 635.20 | 2025-01-01 13:15:00 | 646.05 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-20 12:00:00 | 633.80 | 2025-01-01 13:15:00 | 646.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-12-20 13:00:00 | 634.95 | 2025-01-02 11:15:00 | 650.75 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-12-24 12:00:00 | 634.35 | 2025-01-02 11:15:00 | 650.75 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-12-24 15:15:00 | 629.90 | 2025-01-02 11:15:00 | 650.75 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2024-12-31 12:00:00 | 635.20 | 2025-01-02 11:15:00 | 650.75 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-01-08 09:15:00 | 627.30 | 2025-01-09 09:15:00 | 652.70 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-02-07 11:30:00 | 654.20 | 2025-02-07 13:15:00 | 645.95 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-21 10:00:00 | 628.25 | 2025-03-21 15:15:00 | 632.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-03-21 10:30:00 | 629.05 | 2025-03-21 15:15:00 | 632.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-03-25 09:15:00 | 626.40 | 2025-03-25 09:15:00 | 632.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-25 11:15:00 | 627.60 | 2025-03-26 09:15:00 | 636.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-03-25 13:15:00 | 625.05 | 2025-03-26 09:15:00 | 636.65 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-03-25 14:15:00 | 625.20 | 2025-03-26 09:15:00 | 636.65 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-24 09:15:00 | 700.00 | 2025-08-08 13:15:00 | 706.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-07-28 09:15:00 | 698.75 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-07-28 14:15:00 | 695.35 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-07-29 12:45:00 | 695.20 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-07-31 11:15:00 | 712.20 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-31 13:15:00 | 712.10 | 2025-08-29 09:15:00 | 711.80 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-01 09:15:00 | 721.50 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-08-04 10:00:00 | 712.60 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-08-07 15:00:00 | 718.75 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 719.50 | 2025-09-22 13:15:00 | 716.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-08-18 12:30:00 | 719.25 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-19 11:30:00 | 720.40 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-29 09:15:00 | 720.00 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-08-29 11:00:00 | 717.75 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-09-17 10:15:00 | 717.10 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-09-17 10:45:00 | 718.50 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-09-22 10:15:00 | 723.60 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2025-11-13 13:00:00 | 725.40 | 2025-11-28 13:15:00 | 718.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-11-14 13:00:00 | 728.00 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-14 14:00:00 | 727.80 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-27 12:00:00 | 725.00 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-11-27 15:00:00 | 727.95 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-05 11:30:00 | 725.90 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-09 10:00:00 | 726.85 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-12-09 11:00:00 | 727.35 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-11 10:30:00 | 727.00 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-11 15:15:00 | 727.10 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-12 11:15:00 | 726.25 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 12:00:00 | 727.00 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-12 14:30:00 | 726.65 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-12 15:00:00 | 727.25 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-12-15 09:15:00 | 727.05 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-01-28 10:45:00 | 727.35 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-03-12 12:00:00 | 761.65 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-03-12 12:45:00 | 762.35 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-03-12 13:15:00 | 762.30 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-03-13 09:15:00 | 764.00 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-04-13 15:00:00 | 753.10 | 2026-04-20 09:15:00 | 759.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-16 09:30:00 | 748.25 | 2026-04-20 09:15:00 | 759.20 | STOP_HIT | 1.00 | -1.46% |
