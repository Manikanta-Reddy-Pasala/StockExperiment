# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1051.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 130 |
| ALERT1 | 101 |
| ALERT2 | 96 |
| ALERT2_SKIP | 56 |
| ALERT3 | 272 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 138 |
| PARTIAL | 24 |
| TARGET_HIT | 9 |
| STOP_HIT | 130 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 163 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 101
- **Target hits / Stop hits / Partials:** 9 / 130 / 24
- **Avg / median % per leg:** 1.16% / -0.59%
- **Sum % (uncompounded):** 188.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 12 | 16.2% | 4 | 70 | 0 | -0.26% | -18.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 74 | 12 | 16.2% | 4 | 70 | 0 | -0.26% | -18.9% |
| SELL (all) | 89 | 50 | 56.2% | 5 | 60 | 24 | 2.33% | 207.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.37% | -1.4% |
| SELL @ 3rd Alert (retest2) | 88 | 50 | 56.8% | 5 | 59 | 24 | 2.37% | 208.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.37% | -1.4% |
| retest2 (combined) | 162 | 62 | 38.3% | 9 | 129 | 24 | 1.17% | 190.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 622.25 | 629.58 | 630.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 11:15:00 | 620.75 | 627.81 | 629.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 13:15:00 | 628.20 | 627.61 | 628.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 13:15:00 | 628.20 | 627.61 | 628.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 628.20 | 627.61 | 628.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:45:00 | 629.35 | 627.61 | 628.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 633.30 | 628.75 | 629.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 634.60 | 628.75 | 629.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 634.50 | 629.90 | 629.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 635.80 | 631.50 | 630.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 634.10 | 635.49 | 633.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 634.10 | 635.49 | 633.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 630.95 | 634.59 | 633.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:00:00 | 630.95 | 634.59 | 633.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 625.20 | 632.71 | 632.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:00:00 | 625.20 | 632.71 | 632.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 12:15:00 | 621.70 | 630.51 | 631.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 13:15:00 | 617.75 | 627.96 | 630.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 622.00 | 621.80 | 625.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:30:00 | 622.00 | 621.80 | 625.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 624.75 | 622.34 | 624.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 624.75 | 622.34 | 624.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 625.95 | 623.06 | 624.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:00:00 | 625.95 | 623.06 | 624.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 623.00 | 623.05 | 624.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 621.15 | 624.26 | 624.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 15:15:00 | 619.00 | 615.64 | 615.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 15:15:00 | 619.00 | 615.64 | 615.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 620.40 | 617.45 | 616.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 636.00 | 636.53 | 630.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 12:00:00 | 636.00 | 636.53 | 630.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 636.15 | 637.32 | 632.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:30:00 | 645.80 | 638.66 | 633.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 640.70 | 642.87 | 640.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:00:00 | 642.15 | 647.28 | 646.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 632.00 | 644.22 | 645.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 632.00 | 644.22 | 645.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 618.05 | 638.99 | 642.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 636.95 | 634.06 | 638.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 636.95 | 634.06 | 638.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 648.15 | 636.88 | 639.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 648.15 | 636.88 | 639.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 645.35 | 638.58 | 639.91 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 652.80 | 641.42 | 641.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 667.40 | 646.62 | 643.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 13:15:00 | 658.55 | 660.21 | 653.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:00:00 | 658.55 | 660.21 | 653.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 661.20 | 660.41 | 654.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 655.15 | 660.41 | 654.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 656.25 | 663.38 | 660.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 656.25 | 663.38 | 660.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 649.85 | 660.68 | 659.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 649.85 | 660.68 | 659.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 12:15:00 | 653.30 | 657.81 | 657.99 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 14:15:00 | 672.55 | 660.73 | 659.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 10:15:00 | 672.85 | 665.41 | 662.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 671.45 | 671.61 | 667.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 672.75 | 671.61 | 667.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 668.60 | 670.47 | 667.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:45:00 | 666.85 | 670.47 | 667.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 668.70 | 670.11 | 667.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:45:00 | 666.45 | 670.11 | 667.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 666.00 | 669.29 | 667.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 677.40 | 669.29 | 667.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:30:00 | 670.60 | 669.18 | 668.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 670.15 | 669.18 | 668.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:45:00 | 669.80 | 669.31 | 668.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 669.60 | 669.37 | 668.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:45:00 | 669.20 | 669.37 | 668.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 667.10 | 668.92 | 668.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 667.10 | 668.92 | 668.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 668.70 | 668.87 | 668.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 668.80 | 668.87 | 668.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 665.85 | 668.27 | 668.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 663.40 | 668.27 | 668.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-14 10:15:00 | 664.70 | 667.56 | 667.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 664.70 | 667.56 | 667.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 661.00 | 665.64 | 666.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 665.95 | 663.97 | 665.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 665.95 | 663.97 | 665.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 665.95 | 663.97 | 665.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:45:00 | 665.85 | 663.97 | 665.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 665.05 | 664.18 | 665.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 665.10 | 664.18 | 665.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 669.65 | 665.28 | 665.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 669.65 | 665.28 | 665.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 670.00 | 666.22 | 666.19 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 15:15:00 | 665.00 | 666.09 | 666.16 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 672.20 | 667.31 | 666.71 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 653.90 | 664.88 | 665.86 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 15:15:00 | 668.00 | 666.11 | 665.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 10:15:00 | 669.95 | 667.25 | 666.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 13:15:00 | 667.70 | 667.77 | 666.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 667.70 | 667.77 | 666.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 667.70 | 667.77 | 666.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:15:00 | 666.25 | 667.77 | 666.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 667.10 | 667.63 | 666.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:15:00 | 666.00 | 667.63 | 666.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 666.00 | 667.31 | 666.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 644.15 | 667.31 | 666.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 653.00 | 664.45 | 665.62 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 670.30 | 666.99 | 666.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 679.80 | 669.84 | 668.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 674.50 | 676.18 | 672.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 11:30:00 | 677.30 | 676.18 | 672.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 688.55 | 681.00 | 676.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 692.50 | 681.00 | 676.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:45:00 | 692.45 | 685.62 | 679.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 693.30 | 690.94 | 684.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 665.90 | 684.27 | 683.15 | SL hit (close<static) qty=1.00 sl=674.75 alert=retest2 |

### Cycle 17 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 669.95 | 681.41 | 681.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 10:15:00 | 662.40 | 668.37 | 671.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 672.30 | 667.19 | 669.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 672.30 | 667.19 | 669.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 672.30 | 667.19 | 669.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 672.30 | 667.19 | 669.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 677.15 | 669.18 | 670.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 670.10 | 669.24 | 670.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 671.10 | 670.34 | 670.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 13:15:00 | 673.45 | 671.29 | 671.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 673.45 | 671.29 | 671.01 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 658.60 | 669.05 | 670.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 653.00 | 665.84 | 668.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 632.65 | 632.16 | 639.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 632.65 | 632.16 | 639.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 633.65 | 631.12 | 635.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 635.15 | 631.12 | 635.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 640.25 | 633.41 | 636.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 640.25 | 633.41 | 636.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 643.70 | 635.47 | 636.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 643.70 | 635.47 | 636.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 647.05 | 638.91 | 638.19 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 12:15:00 | 636.10 | 639.42 | 639.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 14:15:00 | 634.45 | 638.07 | 639.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 636.75 | 636.37 | 637.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 12:30:00 | 635.70 | 636.37 | 637.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 632.80 | 632.42 | 634.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 634.60 | 632.42 | 634.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 634.05 | 632.74 | 634.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 634.05 | 632.74 | 634.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 638.40 | 633.77 | 634.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 638.40 | 633.77 | 634.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 647.80 | 636.58 | 635.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 14:15:00 | 655.05 | 644.65 | 640.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 650.15 | 650.30 | 645.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 650.15 | 650.30 | 645.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 650.15 | 650.30 | 645.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 652.00 | 650.30 | 645.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 652.10 | 653.42 | 649.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:15:00 | 649.00 | 653.42 | 649.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 652.60 | 653.26 | 649.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 648.05 | 653.26 | 649.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 649.00 | 654.41 | 651.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 645.55 | 654.41 | 651.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 647.65 | 653.06 | 651.13 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 635.40 | 647.18 | 648.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 631.60 | 644.07 | 647.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 650.65 | 642.93 | 645.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 10:15:00 | 650.65 | 642.93 | 645.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 650.65 | 642.93 | 645.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 650.65 | 642.93 | 645.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 643.55 | 643.06 | 645.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 12:15:00 | 642.90 | 643.06 | 645.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 15:15:00 | 650.90 | 646.84 | 646.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 650.90 | 646.84 | 646.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 653.35 | 648.14 | 647.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 649.80 | 650.24 | 648.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 13:15:00 | 649.80 | 650.24 | 648.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 649.80 | 650.24 | 648.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:30:00 | 649.60 | 650.24 | 648.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 649.65 | 650.13 | 648.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:45:00 | 648.00 | 650.13 | 648.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 650.80 | 652.29 | 650.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 650.80 | 652.29 | 650.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 652.00 | 652.23 | 651.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 642.95 | 652.23 | 651.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 643.85 | 650.56 | 650.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 643.00 | 650.56 | 650.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 644.00 | 649.24 | 649.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 11:15:00 | 642.05 | 647.81 | 649.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 14:15:00 | 646.05 | 644.87 | 647.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 646.05 | 644.87 | 647.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 646.00 | 645.10 | 647.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 655.50 | 645.10 | 647.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 653.60 | 646.80 | 647.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 655.55 | 646.80 | 647.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 651.85 | 647.81 | 648.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:45:00 | 649.15 | 647.92 | 648.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 11:45:00 | 645.50 | 644.72 | 646.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 616.69 | 623.17 | 627.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 613.23 | 623.17 | 627.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 614.15 | 612.74 | 618.21 | SL hit (close>ema200) qty=0.50 sl=612.74 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 613.20 | 611.45 | 611.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 622.55 | 614.19 | 612.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 617.45 | 619.43 | 617.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 617.45 | 619.43 | 617.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 617.45 | 619.43 | 617.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 616.60 | 619.43 | 617.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 617.65 | 619.08 | 617.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 617.65 | 619.08 | 617.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 621.75 | 619.61 | 617.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 623.50 | 620.79 | 618.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-30 14:15:00 | 685.85 | 667.27 | 653.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 712.55 | 720.09 | 720.35 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 723.55 | 719.71 | 719.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 14:15:00 | 725.55 | 721.86 | 720.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 10:15:00 | 739.90 | 740.57 | 733.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 739.90 | 740.57 | 733.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 733.75 | 738.48 | 733.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:00:00 | 733.75 | 738.48 | 733.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 734.75 | 737.73 | 733.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:45:00 | 730.45 | 737.73 | 733.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 730.65 | 736.32 | 733.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 730.65 | 736.32 | 733.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 732.00 | 735.45 | 733.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 729.10 | 735.45 | 733.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 732.85 | 734.93 | 733.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:15:00 | 735.45 | 733.61 | 733.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:45:00 | 734.90 | 734.37 | 733.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 736.75 | 734.16 | 733.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 736.40 | 734.69 | 733.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 735.00 | 735.58 | 734.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 739.10 | 735.58 | 734.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 732.35 | 734.88 | 734.49 | SL hit (close<static) qty=1.00 sl=733.95 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 731.00 | 733.76 | 734.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 730.30 | 733.07 | 733.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 734.40 | 732.84 | 733.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 734.40 | 732.84 | 733.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 734.40 | 732.84 | 733.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:15:00 | 735.40 | 732.84 | 733.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 734.00 | 733.07 | 733.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 732.85 | 733.07 | 733.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 731.85 | 732.83 | 733.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 732.15 | 732.83 | 733.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 733.50 | 732.88 | 733.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 734.65 | 732.88 | 733.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 735.50 | 733.40 | 733.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 735.50 | 733.40 | 733.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 731.50 | 733.02 | 733.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 729.60 | 731.60 | 732.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 729.05 | 731.46 | 732.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 728.85 | 731.46 | 732.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 740.00 | 733.59 | 733.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 740.00 | 733.59 | 733.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 745.65 | 738.94 | 736.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 13:15:00 | 737.15 | 740.02 | 737.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 737.15 | 740.02 | 737.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 737.15 | 740.02 | 737.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 737.15 | 740.02 | 737.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 732.05 | 738.43 | 737.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:45:00 | 730.65 | 738.43 | 737.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 736.00 | 737.94 | 737.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 733.95 | 737.94 | 737.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 731.10 | 736.57 | 736.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 717.40 | 728.96 | 732.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 734.00 | 728.53 | 731.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 734.00 | 728.53 | 731.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 734.00 | 728.53 | 731.25 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 13:15:00 | 736.65 | 732.70 | 732.53 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 731.60 | 732.43 | 732.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 725.35 | 731.01 | 731.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 732.15 | 731.24 | 731.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 732.15 | 731.24 | 731.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 732.15 | 731.24 | 731.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 730.35 | 731.24 | 731.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 727.55 | 730.50 | 731.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 730.20 | 730.50 | 731.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 730.90 | 730.35 | 731.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 730.20 | 730.35 | 731.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 727.85 | 729.85 | 730.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 722.00 | 729.50 | 730.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 721.70 | 727.13 | 729.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 685.90 | 693.45 | 702.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 685.62 | 693.45 | 702.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-14 13:15:00 | 701.70 | 693.07 | 699.17 | SL hit (close>ema200) qty=0.50 sl=693.07 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 09:15:00 | 621.25 | 615.29 | 615.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 12:15:00 | 625.60 | 619.17 | 617.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 616.00 | 621.07 | 618.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 616.00 | 621.07 | 618.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 616.00 | 621.07 | 618.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 616.00 | 621.07 | 618.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 611.50 | 619.16 | 618.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 611.50 | 619.16 | 618.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 607.80 | 616.89 | 617.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 12:15:00 | 605.60 | 614.63 | 616.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 614.50 | 614.17 | 615.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 614.50 | 614.17 | 615.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 614.00 | 614.14 | 615.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 608.15 | 614.14 | 615.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 609.45 | 613.20 | 615.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 606.25 | 609.74 | 611.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:45:00 | 605.95 | 609.05 | 611.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 575.94 | 585.91 | 595.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 575.65 | 585.91 | 595.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 578.85 | 577.63 | 583.13 | SL hit (close>ema200) qty=0.50 sl=577.63 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 578.45 | 573.99 | 573.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 590.90 | 579.81 | 577.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 587.35 | 588.46 | 583.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 586.95 | 588.46 | 583.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 589.75 | 588.72 | 584.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:00:00 | 591.60 | 589.29 | 584.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:30:00 | 593.40 | 590.14 | 585.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 593.25 | 590.41 | 586.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 592.35 | 592.19 | 588.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 595.50 | 596.44 | 593.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 595.50 | 596.44 | 593.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 595.90 | 596.33 | 593.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:45:00 | 594.50 | 596.33 | 593.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 590.90 | 595.51 | 594.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 592.65 | 593.24 | 593.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 592.65 | 593.24 | 593.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 586.20 | 591.48 | 592.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 11:15:00 | 592.35 | 591.27 | 592.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 11:15:00 | 592.35 | 591.27 | 592.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 592.35 | 591.27 | 592.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 592.35 | 591.27 | 592.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 590.15 | 591.04 | 591.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 14:30:00 | 589.15 | 590.65 | 591.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 09:45:00 | 589.65 | 590.63 | 591.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 589.35 | 590.63 | 591.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 10:15:00 | 589.10 | 583.57 | 584.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 588.10 | 584.48 | 584.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-03 11:15:00 | 595.35 | 586.65 | 585.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 595.35 | 586.65 | 585.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 599.00 | 595.57 | 592.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 590.50 | 595.49 | 593.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 590.50 | 595.49 | 593.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 590.50 | 595.49 | 593.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 590.50 | 595.49 | 593.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 596.00 | 595.59 | 593.98 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 587.70 | 592.57 | 592.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 587.15 | 591.49 | 592.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 588.35 | 585.79 | 588.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 588.35 | 585.79 | 588.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 588.35 | 585.79 | 588.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 587.70 | 585.79 | 588.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 588.05 | 586.24 | 588.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:45:00 | 588.60 | 586.24 | 588.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 591.60 | 587.31 | 588.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:30:00 | 591.40 | 587.31 | 588.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 590.50 | 587.95 | 588.73 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 591.70 | 589.35 | 589.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 593.90 | 590.70 | 589.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 12:15:00 | 590.60 | 590.68 | 589.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 12:30:00 | 590.60 | 590.68 | 589.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 590.40 | 590.59 | 590.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 590.40 | 590.59 | 590.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 590.85 | 590.64 | 590.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 594.50 | 590.64 | 590.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 587.40 | 591.00 | 590.66 | SL hit (close<static) qty=1.00 sl=589.35 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 581.75 | 589.03 | 589.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 574.85 | 586.19 | 588.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 584.65 | 584.19 | 586.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 584.65 | 584.19 | 586.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 587.80 | 584.91 | 586.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 582.35 | 584.91 | 586.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:45:00 | 583.35 | 584.43 | 586.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 553.23 | 558.41 | 565.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 554.18 | 558.41 | 565.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 553.25 | 551.23 | 557.26 | SL hit (close>ema200) qty=0.50 sl=551.23 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 555.25 | 550.69 | 550.33 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 548.40 | 550.02 | 550.22 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 553.20 | 550.61 | 550.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 557.25 | 552.57 | 551.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 554.05 | 554.23 | 552.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 12:45:00 | 555.20 | 554.23 | 552.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 548.85 | 553.16 | 552.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 548.85 | 553.16 | 552.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 587.75 | 560.07 | 555.49 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 558.40 | 566.80 | 567.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 557.40 | 562.93 | 565.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 566.85 | 563.71 | 565.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 566.85 | 563.71 | 565.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 566.85 | 563.71 | 565.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 568.10 | 563.71 | 565.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 567.75 | 564.52 | 565.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 567.75 | 564.52 | 565.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 568.35 | 565.29 | 566.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 568.20 | 565.29 | 566.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 568.00 | 566.60 | 566.59 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 561.75 | 565.64 | 566.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 559.05 | 563.71 | 565.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 564.70 | 563.91 | 565.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 564.70 | 563.91 | 565.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 567.35 | 564.60 | 565.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 567.35 | 564.60 | 565.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 566.00 | 564.88 | 565.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 567.75 | 564.88 | 565.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 570.20 | 565.94 | 565.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 15:15:00 | 571.60 | 568.75 | 567.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 15:15:00 | 574.50 | 574.56 | 571.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 15:15:00 | 574.50 | 574.56 | 571.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 574.50 | 574.56 | 571.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 566.50 | 574.56 | 571.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 560.50 | 571.75 | 570.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:45:00 | 559.40 | 571.75 | 570.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 559.50 | 569.30 | 569.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 558.25 | 564.73 | 567.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 572.25 | 565.19 | 566.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 572.25 | 565.19 | 566.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 572.25 | 565.19 | 566.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 572.25 | 565.19 | 566.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 577.00 | 567.55 | 567.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 577.00 | 567.55 | 567.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 575.30 | 569.10 | 568.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 578.50 | 573.67 | 570.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 601.00 | 601.16 | 593.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 601.00 | 601.16 | 593.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 598.60 | 599.32 | 595.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 595.25 | 599.32 | 595.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 600.85 | 599.63 | 595.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 602.65 | 600.07 | 596.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:00:00 | 601.45 | 602.22 | 599.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:15:00 | 601.10 | 601.61 | 599.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 12:30:00 | 602.05 | 605.04 | 603.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 601.25 | 604.28 | 602.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:45:00 | 601.20 | 604.28 | 602.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 604.00 | 604.22 | 603.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 15:15:00 | 605.15 | 604.22 | 603.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 598.55 | 603.24 | 602.77 | SL hit (close<static) qty=1.00 sl=601.20 alert=retest2 |

### Cycle 51 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 10:15:00 | 596.35 | 601.86 | 602.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 589.05 | 599.30 | 601.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 595.55 | 585.92 | 590.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 14:15:00 | 595.55 | 585.92 | 590.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 595.55 | 585.92 | 590.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 596.70 | 585.92 | 590.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 588.00 | 586.33 | 590.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 550.45 | 586.33 | 590.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 13:15:00 | 586.85 | 575.51 | 577.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 595.35 | 579.48 | 578.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 595.35 | 579.48 | 578.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 597.60 | 588.80 | 584.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 09:15:00 | 591.15 | 592.97 | 588.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 591.15 | 592.97 | 588.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 591.15 | 592.97 | 588.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 591.15 | 592.97 | 588.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 589.05 | 592.19 | 588.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:30:00 | 587.80 | 592.19 | 588.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 587.70 | 591.29 | 588.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:45:00 | 587.10 | 591.29 | 588.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 602.50 | 593.53 | 589.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 604.45 | 593.53 | 589.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 603.00 | 597.10 | 594.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 603.45 | 599.49 | 596.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:15:00 | 605.10 | 599.49 | 596.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 597.00 | 598.99 | 596.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 599.45 | 598.99 | 596.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 598.45 | 598.88 | 596.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 597.50 | 598.88 | 596.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 606.20 | 600.35 | 597.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:00:00 | 609.05 | 603.35 | 600.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:30:00 | 607.20 | 603.81 | 600.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 15:00:00 | 609.05 | 604.86 | 601.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 610.85 | 605.42 | 602.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 602.20 | 604.77 | 602.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:30:00 | 600.45 | 604.77 | 602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 600.50 | 603.92 | 602.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 600.50 | 603.92 | 602.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 600.60 | 603.26 | 601.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 596.15 | 603.26 | 601.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-04 15:15:00 | 598.30 | 601.23 | 601.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 15:15:00 | 598.30 | 601.23 | 601.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 09:15:00 | 594.60 | 599.91 | 600.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 594.95 | 589.76 | 592.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 594.95 | 589.76 | 592.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 594.95 | 589.76 | 592.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 590.50 | 589.76 | 592.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 600.85 | 591.98 | 593.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 600.85 | 591.98 | 593.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 599.80 | 593.55 | 594.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 595.95 | 593.55 | 594.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 566.15 | 571.06 | 578.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 10:15:00 | 536.36 | 549.45 | 557.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 537.20 | 524.51 | 522.84 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 525.75 | 531.40 | 531.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 524.95 | 528.56 | 530.23 | Break + close below crossover candle low |

### Cycle 56 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 546.70 | 531.49 | 531.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 10:15:00 | 550.05 | 535.20 | 532.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 09:15:00 | 555.20 | 558.00 | 551.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-03 10:00:00 | 555.20 | 558.00 | 551.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 555.70 | 556.84 | 551.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:00:00 | 555.70 | 556.84 | 551.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 549.65 | 556.14 | 553.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 549.65 | 556.14 | 553.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 552.45 | 555.40 | 553.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 549.40 | 555.40 | 553.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 548.45 | 554.01 | 552.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:00:00 | 548.45 | 554.01 | 552.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 550.00 | 553.21 | 552.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 550.00 | 553.21 | 552.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 14:15:00 | 548.50 | 551.69 | 552.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 09:15:00 | 545.65 | 550.04 | 551.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 14:15:00 | 546.35 | 546.19 | 548.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 15:00:00 | 546.35 | 546.19 | 548.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 551.50 | 547.32 | 548.66 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 552.40 | 549.51 | 549.50 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 542.50 | 549.22 | 549.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 13:15:00 | 540.75 | 547.53 | 549.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 10:15:00 | 546.15 | 545.55 | 547.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 10:15:00 | 546.15 | 545.55 | 547.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 546.15 | 545.55 | 547.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:45:00 | 546.00 | 545.55 | 547.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 548.50 | 546.14 | 547.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:30:00 | 549.45 | 546.14 | 547.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 546.45 | 546.20 | 547.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 544.80 | 546.20 | 547.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 504.15 | 496.69 | 500.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 504.15 | 496.69 | 500.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 506.00 | 498.55 | 500.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 514.80 | 498.55 | 500.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 517.40 | 502.32 | 502.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 522.45 | 506.35 | 504.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 12:15:00 | 513.50 | 520.26 | 515.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 12:15:00 | 513.50 | 520.26 | 515.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 513.50 | 520.26 | 515.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 513.50 | 520.26 | 515.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 524.95 | 521.19 | 515.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:15:00 | 530.25 | 521.19 | 515.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 528.85 | 522.04 | 517.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 553.55 | 557.31 | 557.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 553.55 | 557.31 | 557.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 544.00 | 553.11 | 555.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 532.25 | 530.99 | 537.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 532.25 | 530.99 | 537.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 538.30 | 533.04 | 537.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 538.30 | 533.04 | 537.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 539.40 | 534.31 | 537.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 539.40 | 534.31 | 537.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 539.60 | 535.37 | 537.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 551.50 | 535.37 | 537.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 563.80 | 544.16 | 541.62 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 541.75 | 549.34 | 549.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 540.75 | 547.62 | 548.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 547.70 | 545.33 | 547.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 547.70 | 545.33 | 547.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 547.70 | 545.33 | 547.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 542.35 | 545.25 | 546.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 542.70 | 544.74 | 545.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:45:00 | 543.15 | 544.69 | 545.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 553.65 | 546.48 | 546.53 | SL hit (close>static) qty=1.00 sl=552.50 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 556.50 | 548.49 | 547.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 557.30 | 554.01 | 551.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 615.00 | 616.20 | 606.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:45:00 | 616.85 | 616.20 | 606.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 688.10 | 698.05 | 688.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 688.10 | 698.05 | 688.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 686.10 | 695.66 | 688.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 686.70 | 695.66 | 688.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 686.65 | 693.86 | 688.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 12:15:00 | 688.35 | 693.86 | 688.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 13:15:00 | 688.00 | 692.57 | 688.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 14:15:00 | 678.95 | 688.72 | 687.08 | SL hit (close<static) qty=1.00 sl=684.60 alert=retest2 |

### Cycle 65 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 680.60 | 685.06 | 685.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 676.25 | 682.65 | 684.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 12:15:00 | 680.50 | 678.11 | 680.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 12:15:00 | 680.50 | 678.11 | 680.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 680.50 | 678.11 | 680.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 673.85 | 679.73 | 680.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:45:00 | 671.80 | 675.26 | 677.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:00:00 | 674.00 | 674.17 | 676.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 683.30 | 675.99 | 677.47 | SL hit (close>static) qty=1.00 sl=682.50 alert=retest2 |

### Cycle 66 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 687.30 | 679.81 | 679.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 14:15:00 | 687.95 | 682.44 | 680.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 692.95 | 694.35 | 688.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 13:45:00 | 693.05 | 694.35 | 688.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 689.95 | 693.47 | 688.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 689.95 | 693.47 | 688.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 688.00 | 692.38 | 688.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 682.45 | 692.38 | 688.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 684.65 | 690.83 | 688.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:00:00 | 687.15 | 687.20 | 687.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 690.30 | 695.15 | 695.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 690.30 | 695.15 | 695.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 684.75 | 689.88 | 691.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 689.10 | 686.73 | 688.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 689.10 | 686.73 | 688.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 689.10 | 686.73 | 688.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 14:45:00 | 684.30 | 686.33 | 687.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 690.95 | 687.53 | 688.15 | SL hit (close>static) qty=1.00 sl=690.50 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 692.95 | 688.84 | 688.45 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 686.35 | 688.64 | 688.67 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 690.40 | 688.92 | 688.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 697.00 | 690.69 | 689.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 695.10 | 696.23 | 693.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 695.10 | 696.23 | 693.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 695.10 | 696.23 | 693.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 693.45 | 696.23 | 693.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 695.50 | 696.08 | 694.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:30:00 | 695.35 | 696.08 | 694.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 699.35 | 696.73 | 694.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 702.65 | 697.45 | 695.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 700.55 | 700.58 | 698.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:45:00 | 700.25 | 700.76 | 698.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 702.20 | 703.91 | 701.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 700.05 | 703.14 | 701.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 700.05 | 703.14 | 701.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 697.50 | 702.01 | 701.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 697.50 | 702.01 | 701.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-29 14:15:00 | 698.05 | 700.30 | 700.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 14:15:00 | 698.05 | 700.30 | 700.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 694.30 | 698.15 | 699.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 696.90 | 695.68 | 697.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 696.90 | 695.68 | 697.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 696.90 | 695.68 | 697.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 698.00 | 695.68 | 697.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 702.75 | 697.09 | 698.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 702.75 | 697.09 | 698.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 699.75 | 697.63 | 698.17 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 705.10 | 699.12 | 698.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 715.80 | 702.46 | 700.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 13:15:00 | 723.35 | 724.73 | 718.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:30:00 | 720.80 | 724.73 | 718.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 718.00 | 723.30 | 719.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 718.30 | 723.30 | 719.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 715.40 | 721.72 | 719.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 715.40 | 721.72 | 719.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 714.80 | 720.34 | 718.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:30:00 | 714.95 | 720.34 | 718.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 721.60 | 720.31 | 719.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 730.85 | 722.48 | 720.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 769.20 | 770.89 | 770.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 14:15:00 | 769.20 | 770.89 | 770.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 10:15:00 | 766.65 | 769.08 | 769.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 771.70 | 769.60 | 770.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 771.70 | 769.60 | 770.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 771.70 | 769.60 | 770.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 771.70 | 769.60 | 770.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 781.15 | 771.91 | 771.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 10:15:00 | 789.65 | 778.79 | 776.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 792.60 | 794.38 | 787.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 792.60 | 794.38 | 787.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 792.60 | 794.38 | 787.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 789.60 | 794.38 | 787.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 789.05 | 792.49 | 788.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 793.55 | 792.49 | 788.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 786.75 | 790.68 | 788.67 | SL hit (close<static) qty=1.00 sl=788.35 alert=retest2 |

### Cycle 75 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 812.90 | 821.62 | 821.96 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 820.85 | 818.45 | 818.28 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 814.65 | 817.89 | 818.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 811.00 | 815.99 | 817.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 814.15 | 813.21 | 815.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 13:00:00 | 814.15 | 813.21 | 815.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 817.90 | 814.25 | 815.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 817.90 | 814.25 | 815.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 818.10 | 815.02 | 815.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 818.15 | 815.02 | 815.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 820.70 | 816.16 | 816.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 826.95 | 820.45 | 818.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 822.15 | 823.78 | 821.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 822.15 | 823.78 | 821.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 825.00 | 824.02 | 821.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 824.25 | 824.02 | 821.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 822.80 | 825.46 | 823.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 822.80 | 825.46 | 823.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 823.75 | 825.12 | 823.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 822.00 | 825.12 | 823.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 819.25 | 823.95 | 822.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 819.25 | 823.95 | 822.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 823.45 | 823.85 | 822.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 823.80 | 823.81 | 822.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 824.00 | 823.81 | 822.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 809.05 | 819.99 | 821.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 809.05 | 819.99 | 821.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 12:15:00 | 798.85 | 811.64 | 816.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 813.65 | 808.95 | 813.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 813.65 | 808.95 | 813.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 813.65 | 808.95 | 813.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 813.65 | 808.95 | 813.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 817.70 | 810.70 | 813.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 819.90 | 810.70 | 813.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 812.05 | 810.97 | 813.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 809.60 | 810.97 | 813.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 811.00 | 810.15 | 811.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 810.70 | 810.15 | 811.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 805.00 | 810.40 | 811.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 798.70 | 808.06 | 810.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 796.60 | 804.83 | 808.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 769.12 | 783.74 | 792.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 770.45 | 783.74 | 792.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 770.16 | 783.74 | 792.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 764.75 | 783.74 | 792.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 756.77 | 783.74 | 792.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-22 09:15:00 | 729.90 | 755.50 | 771.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 740.25 | 735.89 | 735.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 749.30 | 738.57 | 736.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 740.80 | 743.38 | 740.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 740.80 | 743.38 | 740.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 740.80 | 743.38 | 740.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 750.45 | 743.46 | 742.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 750.55 | 745.01 | 743.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:15:00 | 751.15 | 744.39 | 743.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 742.05 | 745.08 | 745.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 742.05 | 745.08 | 745.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 737.50 | 742.61 | 744.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 737.00 | 735.98 | 739.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 09:30:00 | 731.85 | 734.28 | 738.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 741.85 | 733.79 | 736.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 741.85 | 733.79 | 736.06 | SL hit (close>ema400) qty=1.00 sl=736.06 alert=retest1 |

### Cycle 82 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 755.45 | 739.77 | 738.49 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 730.15 | 739.89 | 740.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 725.70 | 731.27 | 734.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 733.25 | 730.99 | 733.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 733.25 | 730.99 | 733.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 733.25 | 730.99 | 733.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:15:00 | 733.00 | 730.99 | 733.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 732.65 | 731.32 | 733.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 732.65 | 731.32 | 733.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 732.55 | 731.56 | 733.68 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 737.45 | 734.97 | 734.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 747.50 | 738.30 | 736.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 759.70 | 760.03 | 753.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 756.35 | 760.03 | 753.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 751.35 | 758.00 | 755.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 749.80 | 758.00 | 755.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 753.90 | 757.18 | 755.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:00:00 | 755.25 | 756.79 | 755.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 751.85 | 756.78 | 756.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 751.85 | 756.78 | 756.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 744.60 | 753.14 | 754.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 749.00 | 748.62 | 751.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 749.00 | 748.62 | 751.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 752.55 | 749.65 | 751.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 752.65 | 749.65 | 751.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 751.00 | 749.92 | 751.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 751.65 | 749.92 | 751.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 749.65 | 749.87 | 751.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 746.20 | 749.89 | 751.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 747.85 | 748.98 | 750.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 747.10 | 748.98 | 750.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:30:00 | 748.15 | 747.12 | 749.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 736.75 | 743.98 | 747.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 735.05 | 740.19 | 744.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:15:00 | 708.89 | 719.50 | 724.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:15:00 | 710.46 | 719.50 | 724.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:15:00 | 709.75 | 719.50 | 724.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:15:00 | 710.74 | 719.50 | 724.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 712.30 | 711.27 | 717.05 | SL hit (close>ema200) qty=0.50 sl=711.27 alert=retest2 |

### Cycle 86 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 708.15 | 700.39 | 699.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 710.95 | 702.50 | 700.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 710.20 | 711.75 | 709.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 710.20 | 711.75 | 709.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 707.60 | 710.92 | 708.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 707.60 | 710.92 | 708.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 706.25 | 709.98 | 708.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 705.95 | 709.98 | 708.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 701.70 | 707.88 | 707.94 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 712.60 | 708.17 | 707.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 719.35 | 711.34 | 709.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 710.95 | 712.27 | 710.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 710.95 | 712.27 | 710.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 711.40 | 712.10 | 710.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 710.05 | 712.10 | 710.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 709.95 | 711.67 | 710.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 709.95 | 711.67 | 710.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 712.35 | 711.81 | 710.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 706.00 | 711.81 | 710.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 702.70 | 709.98 | 709.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 702.70 | 709.98 | 709.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 699.40 | 707.87 | 708.91 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 711.70 | 708.82 | 708.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 717.30 | 710.52 | 709.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 710.00 | 711.09 | 710.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 710.00 | 711.09 | 710.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 710.00 | 711.09 | 710.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 710.00 | 711.09 | 710.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 718.20 | 712.51 | 710.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 723.05 | 716.33 | 713.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 721.10 | 716.83 | 714.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 709.00 | 714.92 | 714.29 | SL hit (close<static) qty=1.00 sl=709.10 alert=retest2 |

### Cycle 91 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 707.95 | 713.52 | 713.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 706.15 | 712.05 | 713.03 | Break + close below crossover candle low |

### Cycle 92 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 729.45 | 714.22 | 713.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 10:15:00 | 732.05 | 717.78 | 715.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 733.20 | 735.76 | 729.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 730.65 | 735.76 | 729.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 739.35 | 736.48 | 730.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 740.40 | 737.22 | 731.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:00:00 | 743.40 | 740.73 | 740.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 740.00 | 741.03 | 740.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 736.85 | 740.03 | 740.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 736.85 | 740.03 | 740.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 732.05 | 737.54 | 738.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 738.20 | 735.86 | 737.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 738.20 | 735.86 | 737.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 738.20 | 735.86 | 737.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 738.20 | 735.86 | 737.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 734.45 | 735.58 | 737.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 725.35 | 732.64 | 735.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 727.75 | 731.97 | 735.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:15:00 | 727.90 | 731.97 | 735.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:30:00 | 728.30 | 729.76 | 733.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 735.35 | 730.74 | 733.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 739.95 | 733.51 | 734.25 | SL hit (close>static) qty=1.00 sl=739.10 alert=retest2 |

### Cycle 94 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 740.40 | 734.89 | 734.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 740.55 | 736.02 | 735.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 756.90 | 760.48 | 755.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 756.90 | 760.48 | 755.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 756.90 | 760.48 | 755.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:30:00 | 767.50 | 762.55 | 757.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:45:00 | 766.80 | 765.04 | 760.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 767.50 | 765.26 | 761.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 766.95 | 765.26 | 761.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 756.95 | 763.60 | 760.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 756.95 | 763.60 | 760.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 759.25 | 762.73 | 760.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 758.95 | 759.86 | 759.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 758.95 | 759.86 | 759.93 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 762.55 | 760.44 | 760.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 767.85 | 762.25 | 761.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 764.85 | 767.38 | 765.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 764.85 | 767.38 | 765.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 764.85 | 767.38 | 765.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 764.70 | 767.38 | 765.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 772.40 | 768.39 | 766.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 776.10 | 768.39 | 766.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-20 09:15:00 | 853.71 | 803.27 | 796.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 872.80 | 876.14 | 876.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 871.25 | 875.16 | 875.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 871.20 | 871.11 | 873.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 871.20 | 871.11 | 873.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 871.20 | 871.11 | 873.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 871.20 | 871.11 | 873.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 875.00 | 871.89 | 873.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 875.00 | 871.89 | 873.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 880.05 | 873.52 | 873.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 880.05 | 873.52 | 873.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 876.90 | 874.20 | 874.24 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 878.25 | 875.01 | 874.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 880.35 | 876.08 | 875.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 879.95 | 879.97 | 877.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 879.95 | 879.97 | 877.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 883.50 | 880.95 | 878.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 880.25 | 880.95 | 878.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 893.00 | 906.84 | 901.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 894.75 | 906.84 | 901.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 894.35 | 904.34 | 900.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 893.00 | 904.34 | 900.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 14:15:00 | 888.75 | 897.17 | 898.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 882.20 | 893.05 | 895.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 12:15:00 | 888.80 | 885.03 | 888.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 888.80 | 885.03 | 888.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 888.80 | 885.03 | 888.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 888.60 | 885.03 | 888.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 887.90 | 885.61 | 888.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:30:00 | 889.25 | 885.61 | 888.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 886.40 | 885.76 | 888.18 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 889.75 | 888.78 | 888.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 15:15:00 | 891.85 | 889.40 | 889.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 915.75 | 918.22 | 911.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 919.95 | 918.57 | 912.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 919.95 | 918.57 | 912.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 916.80 | 918.57 | 912.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 918.65 | 920.81 | 918.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 918.65 | 920.81 | 918.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 922.80 | 921.21 | 918.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 924.40 | 921.21 | 918.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 915.90 | 920.15 | 918.65 | SL hit (close<static) qty=1.00 sl=916.25 alert=retest2 |

### Cycle 101 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 914.65 | 917.51 | 917.66 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 921.25 | 918.33 | 918.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 10:15:00 | 923.90 | 919.45 | 918.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 944.45 | 948.99 | 942.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 944.45 | 948.99 | 942.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 944.45 | 948.99 | 942.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 943.80 | 948.99 | 942.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 939.80 | 947.15 | 941.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 939.80 | 947.15 | 941.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 944.65 | 946.65 | 942.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 947.25 | 945.83 | 942.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 946.15 | 945.64 | 943.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 940.50 | 949.24 | 950.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 940.50 | 949.24 | 950.39 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 960.70 | 951.18 | 950.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 966.75 | 957.40 | 955.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 981.35 | 988.57 | 978.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 981.35 | 988.57 | 978.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 981.35 | 988.57 | 978.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 982.50 | 988.57 | 978.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 978.55 | 986.57 | 978.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 975.90 | 986.57 | 978.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 980.45 | 985.34 | 978.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 974.05 | 985.34 | 978.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 976.85 | 983.65 | 978.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 976.85 | 983.65 | 978.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 975.00 | 981.92 | 978.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 975.00 | 981.92 | 978.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 973.20 | 980.17 | 977.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 972.45 | 980.17 | 977.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 965.25 | 975.18 | 976.01 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 981.10 | 974.82 | 974.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 09:15:00 | 993.60 | 980.99 | 978.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 985.00 | 988.27 | 984.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 985.00 | 988.27 | 984.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 985.00 | 988.27 | 984.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 983.35 | 988.27 | 984.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 988.80 | 988.03 | 985.40 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 977.00 | 984.58 | 984.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 10:15:00 | 973.65 | 981.10 | 982.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 12:15:00 | 980.20 | 979.90 | 981.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 980.20 | 979.90 | 981.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 981.95 | 980.31 | 981.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:00:00 | 981.95 | 980.31 | 981.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 985.20 | 981.29 | 982.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 985.20 | 981.29 | 982.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 986.00 | 982.23 | 982.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 981.80 | 982.23 | 982.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 977.60 | 979.90 | 981.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 985.25 | 976.61 | 976.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 985.25 | 976.61 | 976.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 987.90 | 978.87 | 977.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 994.55 | 996.79 | 991.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 994.55 | 996.79 | 991.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 996.40 | 996.71 | 992.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 992.25 | 996.71 | 992.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 996.50 | 996.67 | 992.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 1000.90 | 996.32 | 993.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 1002.35 | 997.63 | 994.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1018.25 | 1001.59 | 998.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 1000.95 | 1004.84 | 1005.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1000.95 | 1004.84 | 1005.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 994.50 | 1002.37 | 1003.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 1001.15 | 996.97 | 999.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1001.15 | 996.97 | 999.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1001.15 | 996.97 | 999.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 1001.15 | 996.97 | 999.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1004.10 | 998.39 | 1000.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 1004.10 | 998.39 | 1000.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1003.65 | 999.44 | 1000.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:15:00 | 1004.20 | 999.44 | 1000.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1002.30 | 1000.02 | 1000.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 1000.45 | 1000.10 | 1000.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:00:00 | 1000.55 | 999.96 | 1000.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:30:00 | 999.85 | 999.46 | 1000.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 1005.80 | 1001.14 | 1000.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1005.80 | 1001.14 | 1000.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 1009.25 | 1002.76 | 1001.56 | Break + close above crossover candle high |

### Cycle 111 — SELL (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 09:15:00 | 983.95 | 1000.04 | 1000.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 977.20 | 992.75 | 997.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 12:15:00 | 976.60 | 976.01 | 983.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 13:00:00 | 976.60 | 976.01 | 983.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1006.65 | 983.01 | 984.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1006.65 | 983.01 | 984.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 1003.25 | 987.06 | 986.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 13:15:00 | 1023.05 | 999.90 | 992.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1008.40 | 1015.98 | 1008.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1008.40 | 1015.98 | 1008.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1008.40 | 1015.98 | 1008.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1008.25 | 1015.98 | 1008.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1009.00 | 1014.58 | 1008.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 1012.00 | 1014.58 | 1008.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1005.80 | 1012.82 | 1008.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1005.80 | 1012.82 | 1008.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1003.05 | 1010.87 | 1008.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1003.05 | 1010.87 | 1008.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 1001.50 | 1006.28 | 1006.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 980.10 | 1000.24 | 1003.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1005.00 | 996.69 | 1000.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1005.00 | 996.69 | 1000.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1005.00 | 996.69 | 1000.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1005.00 | 996.69 | 1000.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 988.70 | 995.10 | 999.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 985.65 | 993.68 | 997.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 984.60 | 993.20 | 995.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 989.80 | 972.00 | 970.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 989.80 | 972.00 | 970.33 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 968.95 | 978.54 | 978.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 960.15 | 972.41 | 975.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 967.80 | 967.01 | 971.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 995.20 | 967.01 | 971.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 982.65 | 970.13 | 972.56 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 990.40 | 974.19 | 974.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 995.55 | 988.16 | 985.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 995.20 | 997.40 | 992.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 995.20 | 997.40 | 992.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 995.20 | 997.40 | 992.27 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 987.10 | 994.70 | 995.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 983.85 | 992.53 | 994.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 10:15:00 | 994.00 | 992.82 | 994.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 994.00 | 992.82 | 994.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 994.00 | 992.82 | 994.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 994.00 | 992.82 | 994.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 994.65 | 993.19 | 994.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 996.35 | 993.19 | 994.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 997.00 | 993.95 | 994.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 997.00 | 993.95 | 994.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 998.80 | 994.92 | 994.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 14:15:00 | 1000.60 | 996.06 | 995.35 | Break + close above crossover candle high |

### Cycle 119 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 987.60 | 994.88 | 994.97 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 1000.90 | 995.73 | 995.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 1003.00 | 997.18 | 996.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 14:15:00 | 995.35 | 997.72 | 996.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 995.35 | 997.72 | 996.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 995.35 | 997.72 | 996.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 995.35 | 997.72 | 996.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 993.50 | 996.88 | 996.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 992.15 | 996.88 | 996.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1007.00 | 999.72 | 997.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:15:00 | 1008.10 | 999.72 | 997.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 1008.95 | 1004.83 | 1001.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 996.95 | 1002.46 | 1001.01 | SL hit (close<static) qty=1.00 sl=997.05 alert=retest2 |

### Cycle 121 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 963.30 | 1013.83 | 1016.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 944.40 | 954.04 | 961.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 948.80 | 946.80 | 953.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 958.90 | 946.80 | 953.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 961.00 | 949.64 | 954.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:15:00 | 963.50 | 949.64 | 954.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 966.35 | 952.98 | 955.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 966.35 | 952.98 | 955.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 969.75 | 958.69 | 957.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 971.90 | 962.07 | 959.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 966.90 | 967.59 | 963.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:00:00 | 966.90 | 967.59 | 963.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 963.20 | 967.81 | 964.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 963.20 | 967.81 | 964.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 964.00 | 967.05 | 964.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 933.30 | 967.05 | 964.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 923.30 | 958.30 | 960.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 921.95 | 951.03 | 957.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 942.40 | 938.06 | 946.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 942.40 | 938.06 | 946.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 942.40 | 938.06 | 946.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 933.65 | 938.44 | 945.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 937.70 | 938.52 | 943.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 937.85 | 938.38 | 943.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 936.50 | 938.50 | 943.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 940.40 | 938.88 | 942.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 932.30 | 937.02 | 941.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 886.97 | 904.91 | 916.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 890.82 | 904.91 | 916.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 890.96 | 904.91 | 916.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 889.67 | 904.91 | 916.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 885.68 | 898.42 | 909.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 884.85 | 883.00 | 894.17 | SL hit (close>ema200) qty=0.50 sl=883.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 912.90 | 897.70 | 895.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 918.45 | 904.24 | 899.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 902.45 | 915.49 | 907.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 902.45 | 915.49 | 907.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 902.45 | 915.49 | 907.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 902.05 | 915.49 | 907.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 903.65 | 913.12 | 907.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 903.60 | 913.12 | 907.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 906.95 | 910.41 | 907.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 914.30 | 906.85 | 906.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 14:15:00 | 901.80 | 908.33 | 907.68 | SL hit (close<static) qty=1.00 sl=903.45 alert=retest2 |

### Cycle 125 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 891.55 | 904.98 | 906.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 871.70 | 898.32 | 903.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 879.80 | 867.40 | 877.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 879.80 | 867.40 | 877.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 879.80 | 867.40 | 877.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 884.40 | 867.40 | 877.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 878.90 | 869.70 | 877.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 878.95 | 869.70 | 877.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 875.40 | 870.84 | 877.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 879.30 | 870.84 | 877.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 880.50 | 872.77 | 877.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 899.25 | 872.77 | 877.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 905.60 | 879.34 | 880.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 904.10 | 879.34 | 880.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 907.40 | 884.95 | 882.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 914.90 | 890.94 | 885.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 888.45 | 900.15 | 893.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 888.45 | 900.15 | 893.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 888.45 | 900.15 | 893.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 888.45 | 900.15 | 893.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 888.75 | 897.87 | 893.02 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 884.45 | 890.49 | 890.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 849.60 | 881.10 | 886.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 868.20 | 855.56 | 867.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 868.20 | 855.56 | 867.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 868.20 | 855.56 | 867.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 833.50 | 870.86 | 871.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 858.40 | 858.50 | 862.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 877.85 | 865.69 | 864.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 877.85 | 865.69 | 864.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 878.50 | 868.25 | 866.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 872.95 | 874.11 | 869.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 872.95 | 874.11 | 869.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 872.95 | 874.11 | 869.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 877.70 | 874.11 | 869.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 880.75 | 875.32 | 871.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 10:15:00 | 965.47 | 905.33 | 887.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 15:15:00 | 1040.50 | 1045.93 | 1046.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 1033.75 | 1043.49 | 1045.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 1033.00 | 1031.19 | 1036.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:00:00 | 1033.00 | 1031.19 | 1036.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1029.80 | 1030.92 | 1036.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:30:00 | 1025.80 | 1031.02 | 1035.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 1022.50 | 1028.85 | 1034.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 1025.50 | 1020.02 | 1024.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 1020.10 | 1022.81 | 1024.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1019.50 | 1012.64 | 1016.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 1023.30 | 1012.64 | 1016.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1009.40 | 1011.99 | 1015.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1008.00 | 1011.99 | 1015.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1024.60 | 1013.10 | 1014.62 | SL hit (close>static) qty=1.00 sl=1019.50 alert=retest2 |

### Cycle 130 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1027.00 | 1015.88 | 1015.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 1029.70 | 1019.38 | 1017.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 09:15:00 | 621.15 | 2024-05-23 15:15:00 | 619.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-05-29 09:30:00 | 645.80 | 2024-06-04 10:15:00 | 632.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-05-30 15:15:00 | 640.70 | 2024-06-04 10:15:00 | 632.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-06-04 10:00:00 | 642.15 | 2024-06-04 10:15:00 | 632.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-06-13 09:15:00 | 677.40 | 2024-06-14 10:15:00 | 664.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-06-13 11:30:00 | 670.60 | 2024-06-14 10:15:00 | 664.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-06-13 12:15:00 | 670.15 | 2024-06-14 10:15:00 | 664.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-06-13 12:45:00 | 669.80 | 2024-06-14 10:15:00 | 664.70 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-26 10:15:00 | 692.50 | 2024-06-27 12:15:00 | 665.90 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2024-06-26 11:45:00 | 692.45 | 2024-06-27 12:15:00 | 665.90 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2024-06-27 09:45:00 | 693.30 | 2024-06-27 12:15:00 | 665.90 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2024-07-05 09:30:00 | 670.10 | 2024-07-05 13:15:00 | 673.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-07-05 12:00:00 | 671.10 | 2024-07-05 13:15:00 | 673.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-07-26 12:15:00 | 642.90 | 2024-07-26 15:15:00 | 650.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-08-01 11:45:00 | 649.15 | 2024-08-12 09:15:00 | 616.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 11:45:00 | 645.50 | 2024-08-12 09:15:00 | 613.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:45:00 | 649.15 | 2024-08-13 10:15:00 | 614.15 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2024-08-02 11:45:00 | 645.50 | 2024-08-13 10:15:00 | 614.15 | STOP_HIT | 0.50 | 4.86% |
| BUY | retest2 | 2024-08-21 14:45:00 | 623.50 | 2024-08-30 14:15:00 | 685.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-23 14:15:00 | 735.45 | 2024-09-25 10:15:00 | 732.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-09-23 14:45:00 | 734.90 | 2024-09-25 12:15:00 | 731.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-09-24 10:15:00 | 736.75 | 2024-09-25 12:15:00 | 731.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-09-24 11:45:00 | 736.40 | 2024-09-25 12:15:00 | 731.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-25 09:15:00 | 739.10 | 2024-09-25 12:15:00 | 731.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-27 14:00:00 | 729.60 | 2024-09-30 10:15:00 | 740.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-09-27 14:45:00 | 729.05 | 2024-09-30 10:15:00 | 740.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-09-27 15:15:00 | 728.85 | 2024-09-30 10:15:00 | 740.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-10-09 10:15:00 | 722.00 | 2024-10-14 09:15:00 | 685.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 721.70 | 2024-10-14 09:15:00 | 685.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 10:15:00 | 722.00 | 2024-10-14 13:15:00 | 701.70 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-10-09 11:45:00 | 721.70 | 2024-10-14 13:15:00 | 701.70 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2024-11-07 10:30:00 | 606.25 | 2024-11-11 09:15:00 | 575.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:45:00 | 605.95 | 2024-11-11 09:15:00 | 575.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 10:30:00 | 606.25 | 2024-11-12 13:15:00 | 578.85 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-11-07 11:45:00 | 605.95 | 2024-11-12 13:15:00 | 578.85 | STOP_HIT | 0.50 | 4.47% |
| BUY | retest2 | 2024-11-21 11:00:00 | 591.60 | 2024-11-26 13:15:00 | 592.65 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-11-21 11:30:00 | 593.40 | 2024-11-26 13:15:00 | 592.65 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-11-21 13:15:00 | 593.25 | 2024-11-26 13:15:00 | 592.65 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-11-22 10:15:00 | 592.35 | 2024-11-26 13:15:00 | 592.65 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-11-27 14:30:00 | 589.15 | 2024-12-03 11:15:00 | 595.35 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-11-28 09:45:00 | 589.65 | 2024-12-03 11:15:00 | 595.35 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-28 10:15:00 | 589.35 | 2024-12-03 11:15:00 | 595.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-12-03 10:15:00 | 589.10 | 2024-12-03 11:15:00 | 595.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-12 09:15:00 | 594.50 | 2024-12-12 13:15:00 | 587.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-12-16 09:15:00 | 582.35 | 2024-12-19 09:15:00 | 553.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:45:00 | 583.35 | 2024-12-19 09:15:00 | 554.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 582.35 | 2024-12-20 09:15:00 | 553.25 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:45:00 | 583.35 | 2024-12-20 09:15:00 | 553.25 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2025-01-20 12:30:00 | 602.65 | 2025-01-23 09:15:00 | 598.55 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-01-21 10:00:00 | 601.45 | 2025-01-23 10:15:00 | 596.35 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-01-21 11:15:00 | 601.10 | 2025-01-23 10:15:00 | 596.35 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-01-22 12:30:00 | 602.05 | 2025-01-23 10:15:00 | 596.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-22 15:15:00 | 605.15 | 2025-01-23 10:15:00 | 596.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-27 09:15:00 | 550.45 | 2025-01-28 13:15:00 | 595.35 | STOP_HIT | 1.00 | -8.16% |
| SELL | retest2 | 2025-01-28 13:15:00 | 586.85 | 2025-01-28 13:15:00 | 595.35 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-01-30 13:15:00 | 604.45 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-01-31 15:15:00 | 603.00 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-02-01 10:30:00 | 603.45 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-02-01 11:15:00 | 605.10 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-02-03 13:00:00 | 609.05 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-02-03 13:30:00 | 607.20 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-02-03 15:00:00 | 609.05 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-02-04 09:30:00 | 610.85 | 2025-02-04 15:15:00 | 598.30 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-02-07 12:15:00 | 595.95 | 2025-02-12 09:15:00 | 566.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 595.95 | 2025-02-14 10:15:00 | 536.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-20 14:15:00 | 530.25 | 2025-03-27 14:15:00 | 553.55 | STOP_HIT | 1.00 | 4.39% |
| BUY | retest2 | 2025-03-21 09:15:00 | 528.85 | 2025-03-27 14:15:00 | 553.55 | STOP_HIT | 1.00 | 4.67% |
| SELL | retest2 | 2025-04-09 10:00:00 | 542.35 | 2025-04-09 12:15:00 | 553.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-09 11:00:00 | 542.70 | 2025-04-09 12:15:00 | 553.65 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-04-09 11:45:00 | 543.15 | 2025-04-09 12:15:00 | 553.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-04-30 12:15:00 | 688.35 | 2025-04-30 14:15:00 | 678.95 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-04-30 13:15:00 | 688.00 | 2025-04-30 14:15:00 | 678.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-06 11:15:00 | 673.85 | 2025-05-07 10:15:00 | 683.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-05-06 14:45:00 | 671.80 | 2025-05-07 10:15:00 | 683.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-05-07 10:00:00 | 674.00 | 2025-05-07 10:15:00 | 683.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-09 15:00:00 | 687.15 | 2025-05-14 11:15:00 | 690.30 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-05-19 14:45:00 | 684.30 | 2025-05-20 09:15:00 | 690.95 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-21 09:15:00 | 683.50 | 2025-05-21 13:15:00 | 692.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-05-21 12:15:00 | 683.25 | 2025-05-21 13:15:00 | 692.95 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-05-27 11:15:00 | 702.65 | 2025-05-29 14:15:00 | 698.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-05-28 09:15:00 | 700.55 | 2025-05-29 14:15:00 | 698.05 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-05-28 09:45:00 | 700.25 | 2025-05-29 14:15:00 | 698.05 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-29 09:45:00 | 702.20 | 2025-05-29 14:15:00 | 698.05 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-06-06 10:30:00 | 730.85 | 2025-06-13 14:15:00 | 769.20 | STOP_HIT | 1.00 | 5.25% |
| BUY | retest2 | 2025-06-20 09:15:00 | 793.55 | 2025-06-20 11:15:00 | 786.75 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-20 14:45:00 | 791.25 | 2025-07-03 09:15:00 | 812.90 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-07-11 13:30:00 | 823.80 | 2025-07-14 09:15:00 | 809.05 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-11 14:15:00 | 824.00 | 2025-07-14 09:15:00 | 809.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-07-15 12:15:00 | 809.60 | 2025-07-21 09:15:00 | 769.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 10:30:00 | 811.00 | 2025-07-21 09:15:00 | 770.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 11:00:00 | 810.70 | 2025-07-21 09:15:00 | 770.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 12:15:00 | 805.00 | 2025-07-21 09:15:00 | 764.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 15:15:00 | 796.60 | 2025-07-21 09:15:00 | 756.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 12:15:00 | 809.60 | 2025-07-22 09:15:00 | 729.90 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2025-07-16 10:30:00 | 811.00 | 2025-07-22 09:15:00 | 729.63 | TARGET_HIT | 0.50 | 10.03% |
| SELL | retest2 | 2025-07-16 11:00:00 | 810.70 | 2025-07-22 11:15:00 | 728.64 | TARGET_HIT | 0.50 | 10.12% |
| SELL | retest2 | 2025-07-16 12:15:00 | 805.00 | 2025-07-22 15:15:00 | 724.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-16 15:15:00 | 796.60 | 2025-07-23 13:15:00 | 733.35 | STOP_HIT | 0.50 | 7.94% |
| BUY | retest2 | 2025-08-01 11:15:00 | 750.45 | 2025-08-05 12:15:00 | 742.05 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-01 12:45:00 | 750.55 | 2025-08-05 12:15:00 | 742.05 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-04 10:15:00 | 751.15 | 2025-08-05 12:15:00 | 742.05 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest1 | 2025-08-07 09:30:00 | 731.85 | 2025-08-07 14:15:00 | 741.85 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-20 12:00:00 | 755.25 | 2025-08-21 14:15:00 | 751.85 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-26 09:15:00 | 746.20 | 2025-09-02 09:15:00 | 708.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 10:30:00 | 747.85 | 2025-09-02 09:15:00 | 710.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 11:00:00 | 747.10 | 2025-09-02 09:15:00 | 709.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:30:00 | 748.15 | 2025-09-02 09:15:00 | 710.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 746.20 | 2025-09-03 09:15:00 | 712.30 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-08-26 10:30:00 | 747.85 | 2025-09-03 09:15:00 | 712.30 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2025-08-26 11:00:00 | 747.10 | 2025-09-03 09:15:00 | 712.30 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2025-08-26 13:30:00 | 748.15 | 2025-09-03 09:15:00 | 712.30 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-08-28 12:45:00 | 735.05 | 2025-09-04 11:15:00 | 698.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 12:45:00 | 735.05 | 2025-09-05 14:15:00 | 693.55 | STOP_HIT | 0.50 | 5.65% |
| BUY | retest2 | 2025-09-19 13:45:00 | 723.05 | 2025-09-22 11:15:00 | 709.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-09-19 14:30:00 | 721.10 | 2025-09-22 11:15:00 | 709.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-09-25 11:30:00 | 740.40 | 2025-09-30 09:15:00 | 736.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-29 13:00:00 | 743.40 | 2025-09-30 09:15:00 | 736.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-29 15:15:00 | 740.00 | 2025-09-30 09:15:00 | 736.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-01 11:30:00 | 725.35 | 2025-10-03 11:15:00 | 739.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-01 12:30:00 | 727.75 | 2025-10-03 11:15:00 | 739.95 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-10-01 13:15:00 | 727.90 | 2025-10-03 11:15:00 | 739.95 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-10-01 14:30:00 | 728.30 | 2025-10-03 11:15:00 | 739.95 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-08 12:30:00 | 767.50 | 2025-10-10 12:15:00 | 758.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-09 10:45:00 | 766.80 | 2025-10-10 12:15:00 | 758.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-09 11:45:00 | 767.50 | 2025-10-10 12:15:00 | 758.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-09 12:15:00 | 766.95 | 2025-10-10 12:15:00 | 758.95 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-14 15:15:00 | 776.10 | 2025-10-20 09:15:00 | 853.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-21 09:15:00 | 924.40 | 2025-11-21 09:15:00 | 915.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-27 14:30:00 | 947.25 | 2025-12-03 10:15:00 | 940.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-28 09:15:00 | 946.15 | 2025-12-03 10:15:00 | 940.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-23 09:15:00 | 981.80 | 2025-12-29 14:15:00 | 985.25 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-12-24 09:45:00 | 977.60 | 2025-12-29 14:15:00 | 985.25 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-01-01 15:15:00 | 1000.90 | 2026-01-07 12:15:00 | 1000.95 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-01-02 10:15:00 | 1002.35 | 2026-01-07 12:15:00 | 1000.95 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-01-05 09:15:00 | 1018.25 | 2026-01-07 12:15:00 | 1000.95 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1000.45 | 2026-01-12 13:15:00 | 1005.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-01-12 10:00:00 | 1000.55 | 2026-01-12 13:15:00 | 1005.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-01-12 10:30:00 | 999.85 | 2026-01-12 13:15:00 | 1005.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-22 10:30:00 | 985.65 | 2026-01-29 13:15:00 | 989.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-23 11:30:00 | 984.60 | 2026-01-29 13:15:00 | 989.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-16 13:15:00 | 1008.10 | 2026-02-17 12:15:00 | 996.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-17 10:15:00 | 1008.95 | 2026-02-17 12:15:00 | 996.95 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-18 10:30:00 | 1010.65 | 2026-02-23 09:15:00 | 963.30 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2026-03-10 12:15:00 | 933.65 | 2026-03-13 09:15:00 | 886.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:45:00 | 937.70 | 2026-03-13 09:15:00 | 890.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 15:00:00 | 937.85 | 2026-03-13 09:15:00 | 890.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:15:00 | 936.50 | 2026-03-13 09:15:00 | 889.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 932.30 | 2026-03-13 13:15:00 | 885.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:15:00 | 933.65 | 2026-03-16 14:15:00 | 884.85 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2026-03-10 13:45:00 | 937.70 | 2026-03-16 14:15:00 | 884.85 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2026-03-10 15:00:00 | 937.85 | 2026-03-16 14:15:00 | 884.85 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2026-03-11 09:15:00 | 936.50 | 2026-03-16 14:15:00 | 884.85 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2026-03-11 11:30:00 | 932.30 | 2026-03-16 14:15:00 | 884.85 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2026-03-20 09:15:00 | 914.30 | 2026-03-20 14:15:00 | 901.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-04-02 09:15:00 | 833.50 | 2026-04-06 12:15:00 | 877.85 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2026-04-02 14:30:00 | 858.40 | 2026-04-06 12:15:00 | 877.85 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-04-07 10:15:00 | 877.70 | 2026-04-08 10:15:00 | 965.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 13:30:00 | 880.75 | 2026-04-08 14:15:00 | 968.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-29 11:30:00 | 1025.80 | 2026-05-06 14:15:00 | 1024.60 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-04-29 13:45:00 | 1022.50 | 2026-05-06 15:15:00 | 1027.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-05-04 10:15:00 | 1025.50 | 2026-05-06 15:15:00 | 1027.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-05-04 12:30:00 | 1020.10 | 2026-05-06 15:15:00 | 1027.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-05-06 11:15:00 | 1008.00 | 2026-05-06 15:15:00 | 1027.00 | STOP_HIT | 1.00 | -1.88% |
