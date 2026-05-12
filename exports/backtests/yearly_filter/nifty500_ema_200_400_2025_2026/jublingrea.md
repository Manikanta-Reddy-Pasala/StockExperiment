# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 743.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 19
- **Target hits / Stop hits / Partials:** 8 / 32 / 17
- **Avg / median % per leg:** 2.79% / 1.93%
- **Sum % (uncompounded):** 159.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 3 | 9 | 0 | 1.05% | 12.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 3 | 9 | 0 | 1.05% | 12.6% |
| SELL (all) | 45 | 35 | 77.8% | 5 | 23 | 17 | 3.25% | 146.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 35 | 77.8% | 5 | 23 | 17 | 3.25% | 146.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 38 | 66.7% | 8 | 32 | 17 | 2.79% | 159.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 687.00 | 681.07 | 681.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 692.50 | 681.26 | 681.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 676.50 | 682.88 | 682.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 675.50 | 682.81 | 681.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 674.60 | 682.81 | 681.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 678.95 | 682.69 | 681.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 678.95 | 682.69 | 681.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 672.20 | 682.59 | 681.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 672.20 | 682.59 | 681.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 682.95 | 682.35 | 681.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:30:00 | 681.90 | 682.35 | 681.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 677.10 | 682.30 | 681.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 675.55 | 682.30 | 681.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 677.40 | 682.25 | 681.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 677.40 | 682.25 | 681.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 680.95 | 682.16 | 681.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 681.50 | 682.16 | 681.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 677.45 | 682.11 | 681.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 678.00 | 682.11 | 681.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 679.05 | 682.08 | 681.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 685.30 | 682.11 | 681.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 14:00:00 | 683.00 | 682.14 | 681.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 675.65 | 682.24 | 681.78 | SL hit (close<static) qty=1.00 sl=676.05 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 701.00 | 744.97 | 745.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 690.10 | 723.49 | 731.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 686.00 | 680.94 | 701.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:30:00 | 683.70 | 680.94 | 701.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 699.55 | 681.19 | 701.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 699.55 | 681.19 | 701.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 699.35 | 681.37 | 701.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 708.15 | 681.37 | 701.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 704.50 | 681.60 | 701.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 704.50 | 681.60 | 701.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 703.65 | 681.82 | 701.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 705.00 | 681.82 | 701.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 701.90 | 682.02 | 701.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 690.45 | 682.02 | 701.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 697.40 | 682.91 | 701.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 692.25 | 683.25 | 701.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 699.05 | 685.22 | 701.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 695.55 | 685.33 | 701.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 694.65 | 685.33 | 701.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:45:00 | 694.95 | 685.61 | 701.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 690.90 | 685.75 | 701.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 693.75 | 684.53 | 697.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 697.50 | 685.30 | 697.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 694.50 | 685.76 | 697.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 694.80 | 685.83 | 697.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 693.00 | 685.99 | 697.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 693.90 | 686.18 | 697.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 14:15:00 | 664.10 | 685.89 | 696.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 15:15:00 | 662.53 | 685.66 | 695.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 655.93 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 657.64 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.92 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 660.20 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 656.35 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.06 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.77 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 660.06 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 658.35 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.20 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 681.55 | 680.58 | 691.53 | SL hit (close>ema200) qty=0.50 sl=680.58 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 707.60 | 698.41 | 698.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 719.00 | 698.70 | 698.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 695.05 | 700.02 | 699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 690.00 | 699.92 | 699.22 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 683.30 | 698.44 | 698.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 680.55 | 698.26 | 698.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 694.65 | 692.36 | 695.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 693.45 | 692.37 | 695.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 692.55 | 693.91 | 695.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 692.65 | 693.90 | 695.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:45:00 | 692.10 | 693.88 | 695.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 692.60 | 693.86 | 695.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 698.70 | 692.78 | 695.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 698.70 | 692.78 | 695.04 | SL hit (close>static) qty=1.00 sl=696.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 710.60 | 697.10 | 697.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 722.70 | 697.49 | 697.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 697.50 | 699.42 | 698.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 698.15 | 699.41 | 698.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:00:00 | 701.05 | 699.42 | 698.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 694.05 | 699.37 | 698.25 | SL hit (close<static) qty=1.00 sl=695.85 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 669.80 | 699.89 | 699.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 667.45 | 699.57 | 699.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 674.05 | 668.58 | 681.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:00:00 | 674.05 | 668.58 | 681.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.50 | 661.35 | 675.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 664.95 | 661.43 | 675.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 663.95 | 661.43 | 675.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 662.90 | 662.04 | 675.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 664.45 | 662.10 | 675.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 669.25 | 662.21 | 675.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 673.80 | 662.21 | 675.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 672.75 | 662.31 | 675.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 665.15 | 662.33 | 675.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.70 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 630.75 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 629.75 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.23 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.89 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-20 12:15:00 | 598.46 | 650.68 | 666.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 717.05 | 623.93 | 623.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 10:15:00 | 738.45 | 625.07 | 624.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 13:00:00 | 706.15 | 2025-05-16 11:15:00 | 687.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-05-23 11:45:00 | 685.30 | 2025-05-27 09:15:00 | 675.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-23 14:00:00 | 683.00 | 2025-05-27 09:15:00 | 675.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-28 09:15:00 | 691.50 | 2025-06-11 13:15:00 | 674.80 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-11 12:00:00 | 682.80 | 2025-06-11 13:15:00 | 674.80 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-23 10:00:00 | 715.80 | 2025-07-03 13:15:00 | 787.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 13:00:00 | 715.15 | 2025-08-11 09:15:00 | 685.75 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-08-19 09:15:00 | 716.45 | 2025-08-19 15:15:00 | 701.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-19 09:45:00 | 719.35 | 2025-08-19 15:15:00 | 701.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-10-14 09:15:00 | 690.45 | 2025-11-06 14:15:00 | 664.10 | PARTIAL | 0.50 | 3.82% |
| SELL | retest2 | 2025-10-15 09:15:00 | 697.40 | 2025-11-06 15:15:00 | 662.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 11:45:00 | 692.25 | 2025-11-07 13:15:00 | 655.93 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2025-10-17 10:00:00 | 699.05 | 2025-11-07 13:15:00 | 657.64 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2025-10-17 11:15:00 | 694.65 | 2025-11-07 13:15:00 | 659.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 14:45:00 | 694.95 | 2025-11-07 13:15:00 | 660.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 09:15:00 | 690.90 | 2025-11-07 13:15:00 | 656.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 693.75 | 2025-11-07 13:15:00 | 659.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 694.50 | 2025-11-07 13:15:00 | 659.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:30:00 | 694.80 | 2025-11-07 13:15:00 | 660.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 09:45:00 | 693.00 | 2025-11-07 13:15:00 | 658.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:15:00 | 693.90 | 2025-11-07 13:15:00 | 659.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-14 09:15:00 | 690.45 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-10-15 09:15:00 | 697.40 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2025-10-15 11:45:00 | 692.25 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-10-17 10:00:00 | 699.05 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-10-17 11:15:00 | 694.65 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-10-17 14:45:00 | 694.95 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-10-20 09:15:00 | 690.90 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-10-29 09:45:00 | 693.75 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-10-31 09:45:00 | 694.50 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-10-31 10:30:00 | 694.80 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2025-11-03 09:45:00 | 693.00 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-11-03 14:15:00 | 693.90 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2025-12-17 10:30:00 | 692.55 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-17 12:00:00 | 692.65 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-17 13:45:00 | 692.10 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-17 14:30:00 | 692.60 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-19 12:30:00 | 697.05 | 2025-12-19 14:15:00 | 705.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-12-19 13:45:00 | 697.35 | 2025-12-19 14:15:00 | 705.65 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-19 14:30:00 | 696.25 | 2025-12-19 15:15:00 | 707.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-30 14:00:00 | 701.05 | 2025-12-30 14:15:00 | 694.05 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-31 09:15:00 | 700.40 | 2026-01-05 10:15:00 | 770.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 10:15:00 | 704.45 | 2026-01-05 10:15:00 | 774.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-09 09:45:00 | 702.80 | 2026-01-09 11:15:00 | 692.25 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-10 11:30:00 | 664.95 | 2026-02-16 09:15:00 | 631.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:00:00 | 663.95 | 2026-02-16 09:15:00 | 630.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 662.90 | 2026-02-16 09:15:00 | 629.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 664.45 | 2026-02-16 09:15:00 | 631.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:30:00 | 665.15 | 2026-02-16 09:15:00 | 631.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 664.95 | 2026-02-20 12:15:00 | 598.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-10 12:00:00 | 663.95 | 2026-02-20 12:15:00 | 597.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 662.90 | 2026-02-20 12:15:00 | 598.01 | TARGET_HIT | 0.50 | 9.79% |
| SELL | retest2 | 2026-02-12 11:15:00 | 664.45 | 2026-02-20 12:15:00 | 598.63 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2026-02-12 14:30:00 | 665.15 | 2026-02-20 13:15:00 | 596.61 | TARGET_HIT | 0.50 | 10.30% |
| SELL | retest2 | 2026-04-15 14:15:00 | 668.70 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-04-16 09:45:00 | 668.50 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-16 11:45:00 | 668.80 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.79% |
