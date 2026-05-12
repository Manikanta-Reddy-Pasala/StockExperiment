# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1949.90
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
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 36 |
| PARTIAL | 5 |
| TARGET_HIT | 14 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 26
- **Target hits / Stop hits / Partials:** 14 / 26 / 5
- **Avg / median % per leg:** 1.80% / -0.76%
- **Sum % (uncompounded):** 81.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 18 | 48.6% | 14 | 19 | 4 | 2.60% | 96.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 29 | 10 | 34.5% | 10 | 19 | 0 | 1.24% | 36.1% |
| SELL (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -1.86% | -14.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 7 | 1 | -1.86% | -14.9% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 37 | 11 | 29.7% | 10 | 26 | 1 | 0.57% | 21.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 10:15:00 | 519.40 | 432.51 | 432.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 13:15:00 | 537.85 | 435.14 | 433.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 500.00 | 500.42 | 481.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:15:00 | 505.30 | 500.42 | 481.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 11:30:00 | 504.20 | 500.52 | 482.31 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:15:00 | 503.70 | 500.58 | 482.70 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:45:00 | 504.50 | 500.61 | 482.81 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 14:15:00 | 529.41 | 504.44 | 489.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 14:15:00 | 528.88 | 504.44 | 489.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 14:15:00 | 529.73 | 504.44 | 489.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 09:15:00 | 530.57 | 504.98 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-08-10 10:15:00 | 555.83 | 508.69 | 493.04 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 14:15:00 | 685.00 | 712.57 | 712.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 10:15:00 | 757.20 | 711.96 | 711.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 12:15:00 | 796.50 | 715.77 | 713.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 13:15:00 | 728.40 | 728.92 | 721.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 15:15:00 | 721.05 | 728.81 | 721.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 721.05 | 728.81 | 721.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 756.90 | 728.81 | 721.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 714.00 | 729.05 | 721.77 | SL hit (close<static) qty=1.00 sl=715.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 13:15:00 | 660.00 | 715.36 | 715.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 14:15:00 | 654.60 | 714.76 | 715.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 688.10 | 684.18 | 697.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 11:30:00 | 691.85 | 684.18 | 697.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 12:15:00 | 696.70 | 684.30 | 697.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:00:00 | 696.70 | 684.30 | 697.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 698.30 | 684.44 | 697.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:30:00 | 701.80 | 684.44 | 697.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 701.95 | 684.62 | 697.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:30:00 | 705.25 | 684.62 | 697.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 708.50 | 684.86 | 697.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 09:15:00 | 699.20 | 684.86 | 697.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 10:15:00 | 716.80 | 686.88 | 698.11 | SL hit (close>static) qty=1.00 sl=715.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 12:15:00 | 724.25 | 703.62 | 703.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 732.10 | 704.44 | 703.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 15:15:00 | 836.00 | 841.27 | 792.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 09:15:00 | 872.00 | 841.27 | 792.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 802.75 | 847.48 | 802.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 802.75 | 847.48 | 802.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 785.20 | 846.86 | 802.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 785.20 | 846.86 | 802.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 795.65 | 846.35 | 802.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 781.60 | 846.35 | 802.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 781.10 | 845.18 | 802.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 781.10 | 845.18 | 802.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 779.05 | 843.88 | 801.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 788.05 | 843.24 | 801.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 786.90 | 840.61 | 801.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:45:00 | 792.75 | 836.12 | 800.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:45:00 | 788.75 | 832.84 | 801.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 800.10 | 832.51 | 801.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 795.15 | 832.51 | 801.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 794.40 | 832.13 | 801.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 794.40 | 832.13 | 801.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 795.50 | 831.77 | 801.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:45:00 | 807.10 | 830.34 | 801.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:45:00 | 804.45 | 830.07 | 801.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:00:00 | 803.45 | 827.74 | 801.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 805.50 | 826.22 | 801.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 804.40 | 827.47 | 807.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 804.40 | 827.47 | 807.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 805.50 | 827.25 | 807.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 802.85 | 827.25 | 807.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 802.15 | 826.78 | 807.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 802.15 | 826.78 | 807.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 786.70 | 826.12 | 807.11 | SL hit (close<static) qty=1.00 sl=794.40 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 1571.50 | 1602.55 | 1602.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 1559.80 | 1601.85 | 1602.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 13:00:00 | 1602.00 | 1597.75 | 1600.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1602.90 | 1597.80 | 1600.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 1602.90 | 1597.80 | 1600.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1605.00 | 1597.88 | 1600.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 1605.30 | 1597.88 | 1600.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 1605.00 | 1597.95 | 1600.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 1622.30 | 1597.95 | 1600.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1657.20 | 1602.46 | 1602.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1685.00 | 1613.69 | 1608.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1628.80 | 1628.96 | 1617.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 1628.80 | 1628.96 | 1617.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1617.30 | 1629.07 | 1618.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 1617.30 | 1629.07 | 1618.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1612.30 | 1628.90 | 1617.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1613.80 | 1628.90 | 1617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1619.30 | 1628.80 | 1617.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 1606.50 | 1628.80 | 1617.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1606.50 | 1628.58 | 1617.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1641.20 | 1628.58 | 1617.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1620.20 | 1628.33 | 1618.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 1625.00 | 1628.24 | 1618.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 1592.80 | 1627.85 | 1617.98 | SL hit (close<static) qty=1.00 sl=1604.30 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2519.00 | 2765.23 | 2765.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 2443.40 | 2730.98 | 2748.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1866.00 | 1813.09 | 2015.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 1866.00 | 1813.09 | 2015.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1908.00 | 1755.58 | 1883.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1908.00 | 1755.58 | 1883.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 2097.00 | 1758.98 | 1884.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 2097.00 | 1758.98 | 1884.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 09:15:00 | 505.30 | 2023-08-07 14:15:00 | 529.41 | PARTIAL | 0.50 | 4.77% |
| BUY | retest1 | 2023-07-25 11:30:00 | 504.20 | 2023-08-07 14:15:00 | 528.88 | PARTIAL | 0.50 | 4.90% |
| BUY | retest1 | 2023-07-26 09:15:00 | 503.70 | 2023-08-07 14:15:00 | 529.73 | PARTIAL | 0.50 | 5.17% |
| BUY | retest1 | 2023-07-26 09:45:00 | 504.50 | 2023-08-08 09:15:00 | 530.57 | PARTIAL | 0.50 | 5.17% |
| BUY | retest1 | 2023-07-25 09:15:00 | 505.30 | 2023-08-10 10:15:00 | 555.83 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2023-07-25 11:30:00 | 504.20 | 2023-08-10 10:15:00 | 554.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2023-07-26 09:15:00 | 503.70 | 2023-08-10 10:15:00 | 554.07 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2023-07-26 09:45:00 | 504.50 | 2023-08-10 10:15:00 | 554.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-08-18 09:15:00 | 499.75 | 2023-08-18 13:15:00 | 492.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-08-18 09:45:00 | 503.00 | 2023-08-18 13:15:00 | 492.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2023-08-22 09:15:00 | 513.00 | 2023-09-01 10:15:00 | 564.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-07 09:15:00 | 756.90 | 2024-03-11 11:15:00 | 714.00 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2024-04-02 09:15:00 | 699.20 | 2024-04-03 10:15:00 | 716.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-04-12 15:00:00 | 701.75 | 2024-04-18 11:15:00 | 715.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-04-15 09:15:00 | 687.20 | 2024-04-18 14:15:00 | 666.66 | PARTIAL | 0.50 | 2.99% |
| SELL | retest2 | 2024-04-15 09:15:00 | 687.20 | 2024-04-23 11:15:00 | 695.30 | STOP_HIT | 0.50 | -1.18% |
| SELL | retest2 | 2024-04-16 10:00:00 | 698.40 | 2024-04-24 10:15:00 | 711.30 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-04-16 12:15:00 | 695.55 | 2024-04-24 10:15:00 | 711.30 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-04-18 13:45:00 | 689.15 | 2024-04-24 11:15:00 | 720.50 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2024-04-23 14:30:00 | 695.10 | 2024-04-24 11:15:00 | 720.50 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-06-05 11:15:00 | 788.05 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-06-06 09:15:00 | 786.90 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-06-07 09:45:00 | 792.75 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-11 09:45:00 | 788.75 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-06-12 09:45:00 | 807.10 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2024-06-12 10:45:00 | 804.45 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.58% |
| BUY | retest2 | 2024-06-13 15:00:00 | 803.45 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.46% |
| BUY | retest2 | 2024-06-14 15:00:00 | 805.50 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.70% |
| BUY | retest2 | 2024-08-13 14:15:00 | 847.95 | 2024-08-14 09:15:00 | 820.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-08-16 09:30:00 | 861.90 | 2024-08-23 09:15:00 | 948.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 09:30:00 | 849.05 | 2024-09-05 10:15:00 | 933.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 15:00:00 | 848.60 | 2024-09-05 10:15:00 | 933.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-03 12:15:00 | 890.65 | 2024-09-12 09:15:00 | 977.08 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2024-09-03 13:30:00 | 888.25 | 2024-09-13 11:15:00 | 979.72 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2024-09-03 14:15:00 | 890.45 | 2024-09-13 11:15:00 | 979.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-03 15:15:00 | 890.00 | 2024-09-13 11:15:00 | 979.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 10:45:00 | 921.00 | 2024-10-17 12:15:00 | 1013.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1641.20 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1620.20 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-25 15:15:00 | 1625.00 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1631.60 | 2025-07-08 12:15:00 | 1794.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-22 14:45:00 | 2907.00 | 2025-12-23 13:15:00 | 2776.90 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2026-01-02 10:00:00 | 2945.00 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -6.61% |
| BUY | retest2 | 2026-01-02 12:15:00 | 2890.70 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2026-01-02 13:00:00 | 2895.40 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -5.01% |
