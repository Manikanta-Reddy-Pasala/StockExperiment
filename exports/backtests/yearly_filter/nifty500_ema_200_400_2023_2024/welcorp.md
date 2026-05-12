# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1282.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 22 |
| ALERT1 | 16 |
| ALERT2 | 15 |
| ALERT2_SKIP | 7 |
| ALERT3 | 82 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 40
- **Target hits / Stop hits / Partials:** 5 / 42 / 4
- **Avg / median % per leg:** -0.49% / -1.94%
- **Sum % (uncompounded):** -25.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 5 | 20.0% | 5 | 20 | 0 | 0.01% | 0.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 5 | 20.0% | 5 | 20 | 0 | 0.01% | 0.4% |
| SELL (all) | 26 | 6 | 23.1% | 0 | 22 | 4 | -0.98% | -25.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 6 | 23.1% | 0 | 22 | 4 | -0.98% | -25.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 51 | 11 | 21.6% | 5 | 42 | 4 | -0.49% | -25.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 514.00 | 540.06 | 540.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 10:15:00 | 500.50 | 537.71 | 538.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 09:15:00 | 537.10 | 533.15 | 536.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 09:15:00 | 537.10 | 533.15 | 536.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 537.10 | 533.15 | 536.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 10:00:00 | 537.10 | 533.15 | 536.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 539.55 | 533.22 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 10:45:00 | 544.00 | 533.22 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 536.25 | 533.25 | 536.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:45:00 | 535.35 | 533.28 | 536.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 15:15:00 | 534.00 | 533.37 | 536.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 10:30:00 | 535.00 | 530.85 | 534.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 12:15:00 | 550.85 | 531.14 | 534.88 | SL hit (close>static) qty=1.00 sl=540.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 585.60 | 538.17 | 538.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 586.55 | 541.23 | 539.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 547.55 | 548.56 | 543.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 547.55 | 548.56 | 543.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 547.55 | 548.56 | 543.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 12:15:00 | 558.40 | 548.70 | 543.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 527.95 | 549.13 | 544.65 | SL hit (close<static) qty=1.00 sl=530.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 15:15:00 | 530.00 | 562.27 | 562.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 09:15:00 | 528.10 | 561.93 | 562.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 550.20 | 546.01 | 552.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 09:45:00 | 550.50 | 546.01 | 552.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 550.60 | 546.06 | 552.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 550.60 | 546.06 | 552.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 550.50 | 546.10 | 552.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 552.00 | 546.10 | 552.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 551.65 | 546.16 | 552.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 554.65 | 546.16 | 552.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 565.00 | 546.39 | 552.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 565.00 | 546.39 | 552.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 563.00 | 546.56 | 552.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 564.25 | 546.71 | 553.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 550.95 | 547.06 | 553.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:45:00 | 546.20 | 547.51 | 552.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 558.15 | 547.83 | 553.02 | SL hit (close>static) qty=1.00 sl=555.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 611.55 | 557.74 | 557.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 617.50 | 559.34 | 558.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 616.10 | 623.95 | 601.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 616.10 | 623.95 | 601.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 610.95 | 623.71 | 601.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:00:00 | 610.95 | 623.71 | 601.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 664.05 | 683.34 | 661.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 659.55 | 683.34 | 661.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 660.05 | 683.11 | 661.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 660.05 | 683.11 | 661.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 659.00 | 682.87 | 661.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 659.00 | 682.87 | 661.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 666.20 | 682.71 | 661.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 675.00 | 681.03 | 661.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-27 09:15:00 | 742.50 | 687.76 | 667.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 726.50 | 756.44 | 756.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 712.00 | 755.02 | 755.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 768.10 | 754.42 | 755.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 768.10 | 754.42 | 755.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 768.10 | 754.42 | 755.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 768.10 | 754.42 | 755.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 768.65 | 754.56 | 755.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 770.35 | 754.56 | 755.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 785.70 | 756.67 | 756.53 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 13:15:00 | 705.85 | 757.00 | 757.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 763.00 | 756.91 | 756.89 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 749.85 | 756.83 | 756.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 734.65 | 756.34 | 756.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 750.75 | 748.00 | 752.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 12:00:00 | 750.75 | 748.00 | 752.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 748.40 | 748.00 | 752.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 749.15 | 748.00 | 752.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 758.50 | 748.11 | 752.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 758.50 | 748.11 | 752.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 758.45 | 748.21 | 752.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 780.65 | 748.21 | 752.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 14:15:00 | 793.50 | 755.86 | 755.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 805.80 | 756.98 | 756.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 802.60 | 814.83 | 792.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:00:00 | 802.60 | 814.83 | 792.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 794.00 | 814.62 | 792.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:45:00 | 803.50 | 814.62 | 792.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 792.55 | 814.40 | 792.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 789.65 | 814.40 | 792.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 802.10 | 814.28 | 792.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 798.70 | 814.28 | 792.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 784.00 | 814.03 | 792.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 10:30:00 | 794.30 | 813.88 | 792.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 12:30:00 | 793.15 | 813.40 | 792.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 14:45:00 | 796.05 | 812.91 | 792.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 13:45:00 | 792.90 | 800.13 | 788.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 779.20 | 799.86 | 789.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 779.20 | 799.86 | 789.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 781.00 | 799.67 | 788.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 786.20 | 798.50 | 788.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 789.70 | 798.35 | 788.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 785.35 | 798.35 | 788.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 787.90 | 798.25 | 788.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 790.05 | 798.17 | 788.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 788.20 | 798.17 | 788.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 786.80 | 798.06 | 788.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 786.65 | 798.06 | 788.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 787.15 | 797.95 | 788.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 787.15 | 797.95 | 788.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 786.00 | 797.83 | 788.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 789.00 | 797.75 | 788.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 792.55 | 797.70 | 788.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:45:00 | 792.45 | 797.70 | 788.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 777.55 | 797.28 | 788.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 777.55 | 797.28 | 788.64 | SL hit (close<static) qty=1.00 sl=778.80 alert=retest2 |

### Cycle 11 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 759.35 | 782.90 | 782.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 749.90 | 781.46 | 782.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 11:15:00 | 775.00 | 775.00 | 778.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 775.00 | 775.00 | 778.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 775.00 | 775.00 | 778.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:30:00 | 775.85 | 775.00 | 778.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 783.80 | 775.01 | 778.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 782.65 | 775.01 | 778.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 788.50 | 775.14 | 778.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 788.95 | 775.14 | 778.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 783.95 | 775.95 | 778.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:45:00 | 785.55 | 775.95 | 778.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 782.65 | 776.02 | 778.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:00:00 | 782.65 | 776.02 | 778.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 785.05 | 776.14 | 778.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:45:00 | 785.00 | 776.14 | 778.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 782.70 | 776.21 | 778.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 778.05 | 776.21 | 778.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 793.00 | 776.40 | 778.98 | SL hit (close>static) qty=1.00 sl=785.75 alert=retest2 |

### Cycle 12 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 897.80 | 781.14 | 780.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 901.60 | 786.55 | 783.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 907.00 | 910.30 | 878.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 899.20 | 910.30 | 878.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 892.50 | 912.59 | 889.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 892.50 | 912.59 | 889.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 884.00 | 912.30 | 889.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 878.00 | 912.30 | 889.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 881.00 | 911.99 | 889.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 876.45 | 911.99 | 889.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 871.55 | 911.59 | 889.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 871.55 | 911.59 | 889.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 897.05 | 911.78 | 892.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 888.85 | 911.78 | 892.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 887.80 | 911.54 | 892.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 882.45 | 911.54 | 892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 875.10 | 911.18 | 892.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 875.10 | 911.18 | 892.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 886.00 | 910.15 | 892.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 887.75 | 910.15 | 892.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 879.20 | 909.31 | 892.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 882.20 | 909.31 | 892.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 883.65 | 895.93 | 887.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 12:45:00 | 887.60 | 895.69 | 887.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:30:00 | 887.40 | 895.61 | 887.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 875.10 | 895.15 | 887.76 | SL hit (close<static) qty=1.00 sl=878.95 alert=retest2 |

### Cycle 13 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 831.75 | 883.52 | 883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 828.40 | 882.97 | 883.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 873.95 | 872.86 | 877.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 873.95 | 872.86 | 877.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 870.25 | 872.83 | 877.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 869.65 | 872.84 | 877.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 881.20 | 872.92 | 877.66 | SL hit (close>static) qty=1.00 sl=880.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 894.90 | 881.17 | 881.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 905.95 | 881.74 | 881.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 882.10 | 882.57 | 881.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 882.10 | 882.57 | 881.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 885.30 | 882.61 | 881.88 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 857.40 | 881.10 | 881.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 845.15 | 879.54 | 880.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 879.40 | 873.21 | 876.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 876.00 | 873.24 | 876.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 872.00 | 873.24 | 876.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:00:00 | 871.40 | 873.18 | 876.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:15:00 | 828.40 | 869.07 | 874.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:15:00 | 827.83 | 869.07 | 874.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 872.70 | 853.84 | 864.12 | SL hit (close>ema200) qty=0.50 sl=853.84 alert=retest2 |

### Cycle 16 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 967.70 | 872.46 | 872.34 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 845.55 | 880.35 | 880.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 837.45 | 879.92 | 880.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 829.50 | 826.09 | 845.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 829.50 | 826.09 | 845.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 799.30 | 765.71 | 794.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 777.30 | 795.29 | 802.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 780.15 | 795.14 | 802.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:00:00 | 780.70 | 794.79 | 802.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 778.65 | 794.31 | 801.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |

### Cycle 18 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 808.35 | 805.07 | 805.06 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 796.25 | 804.96 | 805.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 790.05 | 804.71 | 804.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 818.00 | 805.00 | 805.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 824.00 | 805.31 | 805.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 794.50 | 805.78 | 805.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 795.50 | 805.68 | 805.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 795.50 | 805.68 | 805.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 806.90 | 805.63 | 805.33 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 767.20 | 804.69 | 804.87 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 816.50 | 804.90 | 804.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 828.10 | 805.68 | 805.28 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-22 12:45:00 | 535.35 | 2024-04-01 12:15:00 | 550.85 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-03-22 15:15:00 | 534.00 | 2024-04-01 12:15:00 | 550.85 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-04-01 10:30:00 | 535.00 | 2024-04-01 12:15:00 | 550.85 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-04-15 12:15:00 | 558.40 | 2024-04-19 09:15:00 | 527.95 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2024-04-22 12:00:00 | 557.70 | 2024-05-03 09:15:00 | 613.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 15:00:00 | 561.00 | 2024-05-17 11:15:00 | 617.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 09:15:00 | 572.45 | 2024-05-18 11:15:00 | 629.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-01 09:45:00 | 546.20 | 2024-07-01 14:15:00 | 558.15 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-09-23 09:15:00 | 675.00 | 2024-09-27 09:15:00 | 742.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-21 09:30:00 | 667.90 | 2024-11-25 10:15:00 | 734.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 10:30:00 | 794.30 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-04-07 12:30:00 | 793.15 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-07 14:45:00 | 796.05 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-04-16 13:45:00 | 792.90 | 2025-04-22 14:15:00 | 777.55 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-04-21 09:45:00 | 786.20 | 2025-04-23 10:15:00 | 767.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-04-21 10:30:00 | 789.70 | 2025-04-25 09:15:00 | 758.40 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-04-21 11:15:00 | 785.35 | 2025-04-29 09:15:00 | 767.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-04-21 12:00:00 | 787.90 | 2025-04-29 09:15:00 | 767.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-04-23 09:45:00 | 783.25 | 2025-05-07 13:15:00 | 759.35 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-04-24 09:30:00 | 786.05 | 2025-05-07 13:15:00 | 759.35 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-04-28 09:30:00 | 784.85 | 2025-05-07 13:15:00 | 759.35 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-04-28 10:45:00 | 780.45 | 2025-05-07 13:15:00 | 759.35 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-05-16 15:15:00 | 778.05 | 2025-05-19 09:15:00 | 793.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-20 12:30:00 | 780.10 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-05-27 10:00:00 | 780.60 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-05-27 12:15:00 | 780.85 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-13 12:45:00 | 887.60 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-13 13:30:00 | 887.40 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-08-19 09:15:00 | 887.70 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-19 09:45:00 | 889.00 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-26 12:00:00 | 880.30 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-26 12:45:00 | 880.80 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-08-26 14:00:00 | 880.75 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-09-09 11:30:00 | 869.65 | 2025-09-09 12:15:00 | 881.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-11 11:30:00 | 869.95 | 2025-09-12 15:15:00 | 880.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-11 12:30:00 | 869.40 | 2025-09-12 15:15:00 | 880.45 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-07 11:15:00 | 872.00 | 2025-10-13 09:15:00 | 828.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 15:00:00 | 871.40 | 2025-10-13 09:15:00 | 827.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 11:15:00 | 872.00 | 2025-10-27 09:15:00 | 872.70 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2025-10-07 15:00:00 | 871.40 | 2025-10-27 09:15:00 | 872.70 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-10-27 10:00:00 | 872.70 | 2025-10-28 09:15:00 | 907.50 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-10-27 10:45:00 | 874.20 | 2025-10-28 09:15:00 | 907.50 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-02-20 09:15:00 | 777.30 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2026-02-20 10:00:00 | 780.15 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-20 13:00:00 | 780.70 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2026-02-23 10:30:00 | 778.65 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2026-03-05 11:15:00 | 815.45 | 2026-03-09 09:15:00 | 774.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 814.05 | 2026-03-09 09:15:00 | 773.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:15:00 | 815.45 | 2026-03-10 12:15:00 | 803.00 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2026-03-06 10:45:00 | 814.05 | 2026-03-10 12:15:00 | 803.00 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-03-13 11:00:00 | 807.80 | 2026-03-13 11:15:00 | 808.35 | STOP_HIT | 1.00 | -0.07% |
