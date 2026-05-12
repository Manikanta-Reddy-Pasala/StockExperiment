# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 606.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 31
- **Target hits / Stop hits / Partials:** 5 / 35 / 7
- **Avg / median % per leg:** 0.66% / -0.90%
- **Sum % (uncompounded):** 31.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 2 | 10.5% | 0 | 19 | 0 | -2.04% | -38.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 0 | 19 | 0 | -2.04% | -38.7% |
| SELL (all) | 28 | 14 | 50.0% | 5 | 16 | 7 | 2.50% | 69.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 14 | 50.0% | 5 | 16 | 7 | 2.50% | 69.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 16 | 34.0% | 5 | 35 | 7 | 0.66% | 31.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 809.70 | 852.36 | 852.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 805.40 | 851.49 | 852.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 11:15:00 | 850.00 | 848.08 | 850.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 11:15:00 | 850.00 | 848.08 | 850.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 850.00 | 848.08 | 850.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 850.00 | 848.08 | 850.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 852.25 | 848.12 | 850.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:30:00 | 855.00 | 848.12 | 850.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 855.50 | 848.19 | 850.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:00:00 | 855.50 | 848.19 | 850.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 845.00 | 838.62 | 844.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:15:00 | 849.00 | 838.62 | 844.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 843.05 | 838.67 | 844.78 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 870.65 | 849.47 | 849.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 13:15:00 | 873.90 | 850.19 | 849.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 841.60 | 851.32 | 850.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 841.60 | 851.32 | 850.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 841.60 | 851.32 | 850.39 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 815.80 | 849.42 | 849.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 13:15:00 | 792.40 | 835.89 | 840.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 835.90 | 832.73 | 838.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 835.90 | 832.73 | 838.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 835.90 | 832.73 | 838.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 837.60 | 832.73 | 838.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 866.45 | 833.00 | 838.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:00:00 | 866.45 | 833.00 | 838.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 867.80 | 833.34 | 838.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:45:00 | 868.20 | 833.34 | 838.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 13:15:00 | 872.65 | 843.25 | 843.19 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 812.90 | 842.94 | 843.06 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 15:15:00 | 866.45 | 842.33 | 842.32 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 814.20 | 842.97 | 843.06 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 860.50 | 842.78 | 842.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 864.95 | 843.31 | 843.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 842.55 | 844.76 | 843.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 11:15:00 | 842.55 | 844.76 | 843.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 842.55 | 844.76 | 843.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 835.35 | 844.76 | 843.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 837.85 | 844.69 | 843.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 848.20 | 844.67 | 843.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:45:00 | 851.60 | 871.32 | 859.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 10:30:00 | 849.40 | 871.11 | 859.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 826.50 | 869.34 | 859.01 | SL hit (close<static) qty=1.00 sl=837.40 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 775.95 | 853.44 | 853.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 772.15 | 852.63 | 853.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 806.90 | 806.15 | 824.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:45:00 | 807.70 | 806.15 | 824.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 832.80 | 806.36 | 823.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 832.80 | 806.36 | 823.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 836.95 | 806.66 | 824.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 835.80 | 806.66 | 824.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 824.80 | 807.79 | 824.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 828.45 | 807.79 | 824.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 829.20 | 808.00 | 824.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 827.05 | 808.00 | 824.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 832.15 | 808.24 | 824.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 832.15 | 808.24 | 824.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 819.40 | 809.48 | 824.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:00:00 | 816.65 | 810.33 | 824.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 817.65 | 810.56 | 823.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 825.00 | 810.79 | 823.93 | SL hit (close>static) qty=1.00 sl=824.80 alert=retest2 |

### Cycle 10 — BUY (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 13:15:00 | 868.85 | 832.81 | 832.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 871.05 | 835.22 | 833.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 828.15 | 845.79 | 840.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 828.15 | 845.79 | 840.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 828.15 | 845.79 | 840.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 828.15 | 845.79 | 840.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 832.55 | 845.66 | 839.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 840.80 | 845.66 | 839.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 12:30:00 | 837.65 | 845.19 | 840.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 14:45:00 | 834.55 | 845.03 | 840.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 834.90 | 844.51 | 840.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 819.90 | 843.61 | 839.79 | SL hit (close<static) qty=1.00 sl=824.50 alert=retest2 |

### Cycle 11 — SELL (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 10:15:00 | 810.25 | 836.66 | 836.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 805.05 | 836.34 | 836.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 764.10 | 763.47 | 787.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:00:00 | 764.10 | 763.47 | 787.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 772.00 | 763.60 | 786.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:15:00 | 770.55 | 763.68 | 786.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:45:00 | 767.60 | 763.73 | 786.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:15:00 | 732.02 | 762.74 | 784.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 729.22 | 762.11 | 783.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 693.50 | 757.18 | 780.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 771.45 | 693.21 | 692.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 780.75 | 703.14 | 698.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 824.10 | 824.36 | 790.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 824.10 | 824.36 | 790.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 805.35 | 831.50 | 808.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 801.85 | 831.50 | 808.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 800.80 | 831.20 | 808.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 800.80 | 831.20 | 808.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 796.50 | 830.85 | 808.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 796.50 | 830.85 | 808.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 754.00 | 794.89 | 795.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 752.00 | 794.46 | 794.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 770.65 | 767.23 | 776.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 12:00:00 | 770.65 | 767.23 | 776.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 777.30 | 767.33 | 776.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 777.30 | 767.33 | 776.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 774.15 | 767.40 | 776.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 779.55 | 767.40 | 776.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 775.80 | 767.48 | 776.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 774.80 | 767.48 | 776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 776.10 | 767.57 | 776.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 775.95 | 767.57 | 776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 775.45 | 767.65 | 776.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 773.80 | 767.74 | 776.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 780.80 | 768.03 | 776.73 | SL hit (close>static) qty=1.00 sl=780.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-19 14:45:00 | 848.20 | 2024-10-07 10:15:00 | 826.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-10-04 09:45:00 | 851.60 | 2024-10-07 10:15:00 | 826.50 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-10-04 10:30:00 | 849.40 | 2024-10-07 10:15:00 | 826.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-10-09 09:30:00 | 853.45 | 2024-10-11 11:15:00 | 846.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-10-09 15:15:00 | 856.95 | 2024-10-11 11:15:00 | 846.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-10-10 14:15:00 | 859.90 | 2024-10-11 11:15:00 | 846.80 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-10-11 09:45:00 | 857.25 | 2024-10-21 15:15:00 | 858.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-10-14 12:30:00 | 856.90 | 2024-10-21 15:15:00 | 858.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-10-14 14:15:00 | 860.00 | 2024-10-21 15:15:00 | 858.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-10-15 10:30:00 | 860.60 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-10-15 11:30:00 | 860.40 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-10-15 13:15:00 | 863.00 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-10-18 11:00:00 | 866.15 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-10-21 12:15:00 | 866.30 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-10-21 13:45:00 | 868.25 | 2024-10-22 09:15:00 | 837.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-11-28 11:00:00 | 816.65 | 2024-11-29 11:15:00 | 825.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-11-29 10:00:00 | 817.65 | 2024-11-29 11:15:00 | 825.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-23 09:15:00 | 840.80 | 2024-12-30 14:15:00 | 819.90 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-12-26 12:30:00 | 837.65 | 2024-12-30 14:15:00 | 819.90 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-12-26 14:45:00 | 834.55 | 2024-12-30 14:15:00 | 819.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-12-30 09:15:00 | 834.90 | 2024-12-30 14:15:00 | 819.90 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-02-06 11:15:00 | 770.55 | 2025-02-10 11:15:00 | 732.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:45:00 | 767.60 | 2025-02-10 13:15:00 | 729.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:15:00 | 770.55 | 2025-02-12 09:15:00 | 693.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 11:45:00 | 767.60 | 2025-02-12 09:15:00 | 690.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-20 10:45:00 | 769.80 | 2025-05-22 10:15:00 | 771.45 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-05-21 12:00:00 | 768.75 | 2025-05-22 10:15:00 | 771.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-16 10:45:00 | 773.80 | 2025-09-16 13:15:00 | 780.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-22 14:45:00 | 771.75 | 2025-09-24 14:15:00 | 733.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 766.05 | 2025-09-25 10:15:00 | 727.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:45:00 | 771.75 | 2025-10-15 09:15:00 | 757.05 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2025-09-23 09:15:00 | 766.05 | 2025-10-15 09:15:00 | 757.05 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-10-20 15:15:00 | 773.50 | 2025-10-23 11:15:00 | 780.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-31 11:15:00 | 762.45 | 2025-11-03 10:15:00 | 777.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-31 12:00:00 | 761.55 | 2025-11-03 10:15:00 | 777.80 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-06 10:00:00 | 761.80 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-06 11:15:00 | 762.95 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-07 09:15:00 | 751.00 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-11-07 12:45:00 | 756.50 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-07 13:30:00 | 756.50 | 2025-11-17 09:15:00 | 770.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-11 09:45:00 | 755.85 | 2025-11-17 09:15:00 | 770.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-14 09:30:00 | 762.60 | 2025-11-21 14:15:00 | 724.80 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-11-14 15:15:00 | 763.00 | 2025-11-21 15:15:00 | 723.71 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-11-18 09:15:00 | 762.45 | 2025-11-21 15:15:00 | 724.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:30:00 | 762.60 | 2025-12-08 13:15:00 | 686.66 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2025-11-14 15:15:00 | 763.00 | 2025-12-08 14:15:00 | 685.62 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2025-11-18 09:15:00 | 762.45 | 2025-12-08 14:15:00 | 686.21 | TARGET_HIT | 0.50 | 10.00% |
