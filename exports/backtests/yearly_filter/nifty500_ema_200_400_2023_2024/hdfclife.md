# HDFC Life Insurance Company Ltd. (HDFCLIFE)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 619.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 12 |
| ALERT2_SKIP | 5 |
| ALERT3 | 99 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 85 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 94 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 74
- **Target hits / Stop hits / Partials:** 1 / 85 / 8
- **Avg / median % per leg:** -0.38% / -1.04%
- **Sum % (uncompounded):** -35.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 4 | 7.7% | 0 | 52 | 0 | -1.37% | -71.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 52 | 4 | 7.7% | 0 | 52 | 0 | -1.37% | -71.1% |
| SELL (all) | 42 | 16 | 38.1% | 1 | 33 | 8 | 0.84% | 35.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.11% | -3.1% |
| SELL @ 3rd Alert (retest2) | 41 | 16 | 39.0% | 1 | 32 | 8 | 0.93% | 38.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.11% | -3.1% |
| retest2 (combined) | 93 | 20 | 21.5% | 1 | 84 | 8 | -0.35% | -32.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 609.10 | 634.36 | 634.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 10:15:00 | 606.95 | 634.09 | 634.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 626.65 | 625.76 | 629.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 626.65 | 625.76 | 629.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 626.65 | 625.76 | 629.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:45:00 | 629.00 | 625.76 | 629.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 628.50 | 625.80 | 629.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 621.90 | 625.80 | 629.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 11:45:00 | 622.45 | 625.68 | 628.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 630.65 | 625.55 | 628.79 | SL hit (close>static) qty=1.00 sl=629.90 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 668.10 | 631.47 | 631.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 09:15:00 | 672.95 | 635.43 | 633.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 665.55 | 665.79 | 654.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 14:45:00 | 664.95 | 665.79 | 654.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 652.00 | 665.73 | 654.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 652.00 | 665.73 | 654.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 645.15 | 665.53 | 654.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 645.15 | 665.53 | 654.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 647.50 | 657.28 | 652.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:15:00 | 652.15 | 657.28 | 652.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 645.30 | 657.04 | 652.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:45:00 | 646.05 | 657.04 | 652.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 647.70 | 654.64 | 651.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 647.25 | 654.64 | 651.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 643.60 | 654.06 | 651.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 643.60 | 654.06 | 651.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 646.30 | 652.31 | 650.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:45:00 | 646.70 | 652.31 | 650.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 15:15:00 | 614.95 | 648.71 | 648.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 10:15:00 | 613.15 | 648.01 | 648.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 612.00 | 609.11 | 624.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-07 10:00:00 | 612.00 | 609.11 | 624.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 615.50 | 591.68 | 605.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 13:00:00 | 615.50 | 591.68 | 605.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 611.25 | 591.87 | 605.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 15:15:00 | 608.95 | 592.06 | 605.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 608.50 | 592.62 | 605.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 14:15:00 | 608.65 | 593.04 | 605.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:45:00 | 607.40 | 593.47 | 605.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 610.00 | 593.63 | 605.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:30:00 | 610.00 | 593.63 | 605.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 609.75 | 593.79 | 605.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:45:00 | 610.85 | 593.79 | 605.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-07 13:15:00 | 623.75 | 595.65 | 606.30 | SL hit (close>static) qty=1.00 sl=620.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 14:15:00 | 626.15 | 612.94 | 612.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 10:15:00 | 631.45 | 613.41 | 613.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 619.95 | 620.44 | 617.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 619.95 | 620.44 | 617.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 619.95 | 620.44 | 617.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 10:45:00 | 623.15 | 620.44 | 617.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 609.85 | 620.23 | 617.31 | SL hit (close<static) qty=1.00 sl=617.15 alert=retest2 |

### Cycle 5 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 597.90 | 615.00 | 615.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 13:15:00 | 595.40 | 614.63 | 614.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 576.10 | 575.24 | 588.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 11:00:00 | 576.10 | 575.24 | 588.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 588.75 | 567.00 | 578.55 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 607.25 | 585.04 | 584.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 612.40 | 585.75 | 585.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 10:15:00 | 711.60 | 715.88 | 683.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 711.60 | 715.88 | 683.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 702.25 | 722.21 | 708.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 702.25 | 722.21 | 708.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 702.70 | 722.01 | 708.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 704.65 | 722.01 | 708.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 708.00 | 721.38 | 708.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 715.20 | 721.19 | 708.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 711.25 | 721.81 | 710.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 701.40 | 721.38 | 710.68 | SL hit (close<static) qty=1.00 sl=702.35 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 13:15:00 | 688.35 | 704.86 | 704.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 682.90 | 703.50 | 704.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 652.95 | 625.08 | 646.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 653.30 | 625.36 | 646.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 656.15 | 625.36 | 646.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 641.75 | 626.36 | 646.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:45:00 | 648.15 | 626.36 | 646.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 641.90 | 626.52 | 646.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 645.00 | 626.52 | 646.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 636.55 | 624.55 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 641.00 | 624.55 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 638.90 | 624.69 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 638.90 | 624.69 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 638.60 | 624.83 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:15:00 | 638.30 | 624.83 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 637.65 | 624.96 | 639.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 636.15 | 625.05 | 639.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 636.40 | 625.19 | 639.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 643.00 | 625.50 | 639.28 | SL hit (close>static) qty=1.00 sl=639.85 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 683.80 | 633.64 | 633.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.40 | 647.01 | 640.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.17 | 647.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-07 13:00:00 | 656.90 | 658.17 | 647.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.30 | 776.50 | 756.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 761.50 | 776.34 | 756.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 760.80 | 776.02 | 756.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 761.70 | 776.02 | 756.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 763.25 | 775.90 | 756.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 757.25 | 775.35 | 756.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 757.25 | 775.35 | 756.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 750.80 | 766.27 | 766.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 742.70 | 763.41 | 764.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.98 | 764.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 09:15:00 | 737.15 | 761.98 | 764.01 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 756.55 | 754.21 | 759.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 756.55 | 754.21 | 759.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 759.05 | 754.26 | 759.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 758.25 | 754.26 | 759.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 760.05 | 754.31 | 759.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 760.05 | 754.31 | 759.15 | SL hit (close>ema400) qty=1.00 sl=759.15 alert=retest1 |

### Cycle 10 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 761.10 | 761.97 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 761.10 | 761.96 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 760.85 | 761.96 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 761.25 | 761.96 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 758.55 | 761.96 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 762.01 | 760.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 769.30 | 760.64 | 760.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 768.40 | 760.90 | 760.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 754.75 | 761.58 | 760.78 | SL hit (close<static) qty=1.00 sl=760.75 alert=retest2 |

### Cycle 11 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.02 | 761.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.71 | 760.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 762.00 | 758.12 | 759.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 762.95 | 758.17 | 759.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 762.35 | 758.17 | 759.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 759.85 | 758.29 | 759.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 759.85 | 758.29 | 759.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 760.60 | 758.31 | 759.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 775.20 | 758.31 | 759.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 773.20 | 758.46 | 759.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 754.15 | 760.24 | 760.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 752.60 | 760.17 | 760.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 751.95 | 759.47 | 760.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 751.15 | 758.93 | 759.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 716.44 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 714.97 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 714.35 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 713.59 | 748.09 | 753.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.33 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-14 13:45:00 | 637.30 | 2023-08-16 09:15:00 | 624.95 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2023-08-22 09:15:00 | 638.00 | 2023-08-25 15:15:00 | 627.30 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-08-22 14:45:00 | 635.35 | 2023-08-25 15:15:00 | 627.30 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-08-23 09:15:00 | 635.80 | 2023-08-25 15:15:00 | 627.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-08-30 09:15:00 | 636.60 | 2023-10-04 09:15:00 | 627.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-08-30 09:45:00 | 636.05 | 2023-10-04 09:15:00 | 627.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-08-31 10:00:00 | 637.60 | 2023-10-04 09:15:00 | 627.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-08-31 11:00:00 | 636.15 | 2023-10-04 09:15:00 | 627.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-10-16 11:15:00 | 629.45 | 2023-10-25 09:15:00 | 613.20 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-10-16 13:00:00 | 628.95 | 2023-10-25 09:15:00 | 613.20 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2023-10-16 14:15:00 | 629.05 | 2023-10-25 09:15:00 | 613.20 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2023-11-13 09:15:00 | 621.90 | 2023-11-15 09:15:00 | 630.65 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-11-13 11:45:00 | 622.45 | 2023-11-15 09:15:00 | 630.65 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-03-04 15:15:00 | 608.95 | 2024-03-07 13:15:00 | 623.75 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-03-05 11:00:00 | 608.50 | 2024-03-07 13:15:00 | 623.75 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-03-05 14:15:00 | 608.65 | 2024-03-07 13:15:00 | 623.75 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-03-06 09:45:00 | 607.40 | 2024-03-07 13:15:00 | 623.75 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-04-12 10:45:00 | 623.15 | 2024-04-15 09:15:00 | 609.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-10-28 09:45:00 | 715.20 | 2024-11-05 09:15:00 | 701.40 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-11-04 14:15:00 | 711.25 | 2024-11-05 09:15:00 | 701.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-05 13:30:00 | 711.50 | 2024-11-08 14:15:00 | 708.55 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-11-07 11:30:00 | 711.75 | 2024-11-12 12:15:00 | 699.60 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-11-08 10:15:00 | 719.25 | 2024-11-12 12:15:00 | 699.60 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-01-31 13:45:00 | 636.15 | 2025-02-01 09:15:00 | 643.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-31 14:30:00 | 636.40 | 2025-02-01 09:15:00 | 643.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-02-01 12:30:00 | 633.10 | 2025-02-01 13:15:00 | 601.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 12:30:00 | 633.10 | 2025-02-03 10:15:00 | 626.80 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-02-05 11:00:00 | 635.60 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-02-06 14:45:00 | 633.10 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-02-07 11:45:00 | 634.20 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-02-10 10:15:00 | 633.15 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-02-10 12:45:00 | 634.80 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-10 15:15:00 | 633.00 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-02-12 13:30:00 | 634.50 | 2025-02-28 09:15:00 | 603.82 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-02-13 13:15:00 | 634.80 | 2025-02-28 09:15:00 | 603.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 634.50 | 2025-03-05 11:15:00 | 622.60 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-02-13 13:15:00 | 634.80 | 2025-03-05 11:15:00 | 622.60 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-03-12 09:15:00 | 630.35 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-03-12 15:15:00 | 630.50 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-03-13 10:45:00 | 629.60 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-17 09:45:00 | 629.55 | 2025-03-18 13:15:00 | 638.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-14 10:45:00 | 761.50 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-14 12:30:00 | 760.80 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-14 13:15:00 | 761.70 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-14 14:00:00 | 763.25 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-15 13:30:00 | 757.75 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-15 15:15:00 | 757.90 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-16 14:30:00 | 758.40 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-17 12:30:00 | 757.70 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-22 12:45:00 | 765.05 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-07-23 13:15:00 | 764.15 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-23 13:45:00 | 764.25 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-24 09:30:00 | 764.05 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-30 12:15:00 | 757.60 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-30 13:30:00 | 756.60 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-30 15:00:00 | 758.20 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-31 09:15:00 | 756.95 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-31 12:30:00 | 758.00 | 2025-08-01 11:15:00 | 744.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-08-08 09:15:00 | 759.65 | 2025-09-04 13:15:00 | 752.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-08 10:15:00 | 758.30 | 2025-09-09 09:15:00 | 753.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-08 11:00:00 | 758.70 | 2025-09-09 09:15:00 | 753.55 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-08-11 12:45:00 | 759.65 | 2025-09-25 09:15:00 | 764.55 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-09-05 09:15:00 | 760.20 | 2025-09-25 09:15:00 | 764.55 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-09-05 10:15:00 | 760.00 | 2025-09-26 14:15:00 | 764.45 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-09-09 15:00:00 | 760.10 | 2025-09-26 14:15:00 | 764.45 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-09-18 09:15:00 | 774.90 | 2025-09-30 10:15:00 | 752.35 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-09-24 10:30:00 | 770.45 | 2025-10-08 10:15:00 | 744.90 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-09-26 09:30:00 | 774.90 | 2025-10-08 10:15:00 | 744.90 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-09-26 11:45:00 | 769.60 | 2025-10-08 10:15:00 | 744.90 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest1 | 2025-10-16 09:15:00 | 737.15 | 2025-10-29 12:15:00 | 760.05 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-11-10 14:00:00 | 754.15 | 2025-11-11 11:15:00 | 759.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-11 11:15:00 | 754.90 | 2025-11-11 11:15:00 | 759.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-20 09:30:00 | 750.85 | 2025-11-20 12:15:00 | 759.75 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-05 12:15:00 | 769.30 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-05 15:00:00 | 768.40 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-10 11:00:00 | 769.85 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-11 10:45:00 | 768.35 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-23 10:30:00 | 763.30 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-23 14:45:00 | 763.00 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-24 12:00:00 | 763.15 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-09 09:15:00 | 754.15 | 2026-01-23 09:15:00 | 716.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:15:00 | 752.60 | 2026-01-23 09:15:00 | 714.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 751.95 | 2026-01-23 09:15:00 | 714.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 751.15 | 2026-01-23 14:15:00 | 713.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 754.15 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2026-01-09 11:15:00 | 752.60 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2026-01-13 09:15:00 | 751.95 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-01-13 14:00:00 | 751.15 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2026-02-20 09:15:00 | 727.40 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-20 09:45:00 | 730.75 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-20 13:00:00 | 731.00 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-20 14:00:00 | 730.75 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-27 09:15:00 | 730.05 | 2026-03-04 09:15:00 | 693.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 730.05 | 2026-03-09 09:15:00 | 657.04 | TARGET_HIT | 0.50 | 10.00% |
