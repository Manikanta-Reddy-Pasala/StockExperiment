# Caplin Point Laboratories Ltd. (CAPLIPOINT)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1854.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 228 |
| ALERT1 | 149 |
| ALERT2 | 149 |
| ALERT2_SKIP | 71 |
| ALERT3 | 375 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 219 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 219 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 237 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 59 / 178
- **Target hits / Stop hits / Partials:** 10 / 219 / 8
- **Avg / median % per leg:** -0.30% / -0.97%
- **Sum % (uncompounded):** -71.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 29 | 33.3% | 8 | 79 | 0 | 0.36% | 31.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.57% | -1.6% |
| BUY @ 3rd Alert (retest2) | 86 | 29 | 33.7% | 8 | 78 | 0 | 0.38% | 33.0% |
| SELL (all) | 150 | 30 | 20.0% | 2 | 140 | 8 | -0.69% | -103.1% |
| SELL @ 2nd Alert (retest1) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.22% | -12.2% |
| SELL @ 3rd Alert (retest2) | 140 | 30 | 21.4% | 2 | 130 | 8 | -0.65% | -90.9% |
| retest1 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.25% | -13.8% |
| retest2 (combined) | 226 | 59 | 26.1% | 10 | 208 | 8 | -0.26% | -57.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 692.05 | 706.66 | 708.18 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 711.00 | 706.30 | 705.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 718.15 | 709.72 | 707.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 12:15:00 | 737.70 | 738.67 | 729.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 12:45:00 | 733.00 | 738.67 | 729.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 731.25 | 736.05 | 731.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:00:00 | 731.25 | 736.05 | 731.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 735.75 | 735.99 | 731.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:30:00 | 737.80 | 735.79 | 732.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 756.40 | 735.84 | 733.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 748.15 | 740.94 | 738.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-08 09:15:00 | 811.58 | 784.76 | 776.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 807.10 | 812.20 | 812.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 14:15:00 | 806.50 | 809.75 | 811.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 810.05 | 808.94 | 810.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 810.05 | 808.94 | 810.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 810.05 | 808.94 | 810.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:30:00 | 811.35 | 808.94 | 810.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 803.35 | 807.82 | 809.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 11:15:00 | 800.50 | 807.82 | 809.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:30:00 | 800.70 | 805.79 | 808.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 15:15:00 | 799.60 | 805.79 | 808.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 10:45:00 | 800.30 | 803.41 | 806.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 790.75 | 793.14 | 799.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 802.95 | 795.99 | 795.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 802.95 | 795.99 | 795.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 09:15:00 | 794.00 | 804.85 | 805.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 13:15:00 | 790.05 | 798.39 | 801.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 15:15:00 | 794.00 | 791.06 | 794.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 15:15:00 | 794.00 | 791.06 | 794.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 794.00 | 791.06 | 794.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 791.60 | 791.06 | 794.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 795.00 | 791.85 | 794.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:15:00 | 798.05 | 791.85 | 794.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 795.80 | 792.64 | 795.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 12:30:00 | 793.40 | 793.26 | 794.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 13:30:00 | 792.80 | 793.30 | 794.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 10:30:00 | 790.55 | 793.75 | 794.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 09:15:00 | 802.10 | 791.82 | 792.75 | SL hit (close>static) qty=1.00 sl=801.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 10:15:00 | 805.05 | 794.47 | 793.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 12:15:00 | 813.60 | 800.13 | 796.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 11:15:00 | 812.70 | 813.01 | 805.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 11:30:00 | 813.60 | 813.01 | 805.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 806.60 | 811.00 | 806.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:00:00 | 806.60 | 811.00 | 806.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 813.80 | 811.56 | 806.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 821.45 | 812.45 | 807.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 801.50 | 813.68 | 813.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 801.50 | 813.68 | 813.79 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 818.90 | 814.33 | 813.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 821.05 | 815.75 | 814.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 14:15:00 | 862.65 | 865.25 | 853.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 15:15:00 | 862.05 | 865.25 | 853.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 15:15:00 | 862.05 | 864.61 | 854.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:45:00 | 865.95 | 865.15 | 855.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 11:15:00 | 865.40 | 865.10 | 856.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 12:15:00 | 866.50 | 864.93 | 856.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 14:00:00 | 867.95 | 863.22 | 860.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 863.00 | 863.18 | 861.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:30:00 | 861.70 | 863.18 | 861.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 860.00 | 862.54 | 860.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 856.25 | 862.54 | 860.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 858.55 | 861.75 | 860.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-21 11:15:00 | 853.90 | 859.90 | 860.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 853.90 | 859.90 | 860.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 12:15:00 | 851.00 | 858.12 | 859.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 10:15:00 | 858.90 | 855.90 | 857.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 10:15:00 | 858.90 | 855.90 | 857.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 858.90 | 855.90 | 857.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:15:00 | 860.95 | 855.90 | 857.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 859.05 | 856.53 | 857.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 14:45:00 | 853.85 | 856.69 | 857.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 10:30:00 | 855.85 | 856.92 | 857.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 11:00:00 | 856.90 | 856.92 | 857.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 11:30:00 | 855.65 | 857.29 | 857.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 12:15:00 | 865.10 | 858.85 | 858.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 12:15:00 | 865.10 | 858.85 | 858.16 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 10:15:00 | 851.80 | 857.00 | 857.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 12:15:00 | 843.40 | 853.48 | 855.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 859.70 | 851.00 | 853.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 859.70 | 851.00 | 853.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 859.70 | 851.00 | 853.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 859.70 | 851.00 | 853.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 855.90 | 851.98 | 853.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:30:00 | 859.50 | 851.98 | 853.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 859.90 | 853.89 | 854.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:00:00 | 859.90 | 853.89 | 854.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 856.40 | 854.40 | 854.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:30:00 | 858.90 | 854.40 | 854.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 850.00 | 853.37 | 854.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 864.30 | 853.37 | 854.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 881.20 | 858.94 | 856.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 13:15:00 | 892.00 | 880.29 | 872.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 15:15:00 | 910.00 | 910.57 | 902.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-03 09:15:00 | 921.35 | 910.57 | 902.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 942.95 | 957.23 | 944.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:00:00 | 942.95 | 957.23 | 944.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 879.95 | 941.77 | 938.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 15:00:00 | 879.95 | 941.77 | 938.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 15:15:00 | 890.00 | 931.42 | 934.15 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 974.90 | 940.11 | 937.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 11:15:00 | 983.20 | 954.18 | 944.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 1036.60 | 1046.29 | 1021.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 09:45:00 | 1036.65 | 1046.29 | 1021.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 1028.30 | 1040.26 | 1024.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 12:45:00 | 1024.90 | 1040.26 | 1024.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 1035.00 | 1037.65 | 1027.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 1014.20 | 1037.65 | 1027.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1028.40 | 1035.80 | 1027.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 1018.55 | 1035.80 | 1027.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 1027.00 | 1034.04 | 1027.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:45:00 | 1024.65 | 1034.04 | 1027.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 1022.75 | 1031.78 | 1026.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:30:00 | 1022.90 | 1031.78 | 1026.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 1024.95 | 1030.42 | 1026.66 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 15:15:00 | 1014.85 | 1024.72 | 1024.74 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 1053.00 | 1030.38 | 1027.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 14:15:00 | 1060.45 | 1046.52 | 1037.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 1044.05 | 1050.21 | 1042.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 1044.05 | 1050.21 | 1042.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 1044.05 | 1050.21 | 1042.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 1044.05 | 1050.21 | 1042.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 1042.30 | 1048.63 | 1042.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:00:00 | 1042.30 | 1048.63 | 1042.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 1043.75 | 1047.65 | 1042.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 14:30:00 | 1045.35 | 1048.02 | 1043.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 1033.20 | 1045.37 | 1042.88 | SL hit (close<static) qty=1.00 sl=1039.35 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 13:15:00 | 1043.00 | 1046.15 | 1046.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 14:15:00 | 1034.20 | 1043.76 | 1045.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 12:15:00 | 1036.00 | 1031.61 | 1035.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 12:15:00 | 1036.00 | 1031.61 | 1035.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 1036.00 | 1031.61 | 1035.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 1036.00 | 1031.61 | 1035.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 1038.45 | 1032.98 | 1035.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 1038.45 | 1032.98 | 1035.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 1032.50 | 1032.88 | 1035.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 15:15:00 | 1025.00 | 1032.88 | 1035.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 14:45:00 | 1027.00 | 1031.19 | 1033.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 10:15:00 | 1026.60 | 1029.43 | 1031.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 12:30:00 | 1026.10 | 1025.33 | 1029.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1033.00 | 1026.72 | 1028.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 12:45:00 | 1019.55 | 1025.68 | 1027.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 1048.10 | 1029.94 | 1028.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 1048.10 | 1029.94 | 1028.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 1062.90 | 1036.53 | 1031.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 1071.00 | 1071.52 | 1059.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 09:30:00 | 1071.05 | 1071.52 | 1059.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 1051.75 | 1068.93 | 1062.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 1051.75 | 1068.93 | 1062.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 1058.00 | 1066.74 | 1062.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 1077.35 | 1066.74 | 1062.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 11:15:00 | 1078.00 | 1090.66 | 1092.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 1078.00 | 1090.66 | 1092.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 10:15:00 | 1068.15 | 1077.36 | 1083.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 993.45 | 988.21 | 1011.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 11:45:00 | 993.85 | 988.21 | 1011.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 1003.40 | 993.98 | 1007.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 1011.15 | 993.98 | 1007.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1007.60 | 996.71 | 1007.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 10:45:00 | 991.80 | 997.34 | 1006.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:45:00 | 995.20 | 996.98 | 1005.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:30:00 | 996.95 | 998.29 | 1004.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 14:15:00 | 1030.00 | 1004.63 | 1006.87 | SL hit (close>static) qty=1.00 sl=1019.80 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 1031.50 | 1010.00 | 1009.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 13:15:00 | 1040.00 | 1026.68 | 1018.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 1033.45 | 1041.60 | 1034.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 1033.45 | 1041.60 | 1034.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1033.45 | 1041.60 | 1034.38 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 1021.15 | 1031.11 | 1031.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 1019.40 | 1028.77 | 1030.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 1001.60 | 1000.80 | 1007.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 13:30:00 | 996.55 | 1000.07 | 1006.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 15:00:00 | 995.95 | 999.25 | 1005.98 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 1009.00 | 1000.12 | 1005.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-26 09:15:00 | 1009.00 | 1000.12 | 1005.14 | SL hit (close>ema400) qty=1.00 sl=1005.14 alert=retest1 |

### Cycle 22 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 1015.00 | 1008.43 | 1007.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 09:15:00 | 1038.00 | 1014.35 | 1010.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 13:15:00 | 1022.30 | 1023.71 | 1016.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 14:00:00 | 1022.30 | 1023.71 | 1016.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 1023.10 | 1028.36 | 1023.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 1023.10 | 1028.36 | 1023.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 1036.50 | 1029.99 | 1024.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:30:00 | 1040.00 | 1031.68 | 1028.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 13:00:00 | 1050.05 | 1044.85 | 1039.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 14:15:00 | 1056.60 | 1065.58 | 1065.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 1056.60 | 1065.58 | 1065.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 1050.00 | 1062.46 | 1064.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 1065.35 | 1062.92 | 1064.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 10:15:00 | 1065.35 | 1062.92 | 1064.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 1065.35 | 1062.92 | 1064.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 14:15:00 | 1060.15 | 1063.26 | 1064.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 12:30:00 | 1059.10 | 1062.40 | 1063.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:15:00 | 1060.10 | 1062.40 | 1063.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:45:00 | 1059.25 | 1061.99 | 1062.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 1060.25 | 1061.64 | 1062.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:45:00 | 1065.50 | 1061.64 | 1062.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-11 15:15:00 | 1074.00 | 1064.11 | 1063.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 15:15:00 | 1074.00 | 1064.11 | 1063.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 11:15:00 | 1079.30 | 1068.53 | 1065.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 1106.30 | 1107.99 | 1095.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 09:15:00 | 1084.00 | 1107.99 | 1095.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1088.35 | 1104.06 | 1095.05 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 10:15:00 | 1085.45 | 1092.99 | 1093.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 11:15:00 | 1076.40 | 1089.68 | 1091.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 1082.25 | 1080.23 | 1085.55 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:30:00 | 1067.85 | 1076.95 | 1083.58 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 12:15:00 | 1068.15 | 1076.01 | 1082.55 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1093.00 | 1079.25 | 1082.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 1093.00 | 1079.25 | 1082.40 | SL hit (close>ema400) qty=1.00 sl=1082.40 alert=retest1 |

### Cycle 26 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 1018.00 | 1006.98 | 1006.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 12:15:00 | 1022.05 | 1016.08 | 1012.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 13:15:00 | 1026.00 | 1027.09 | 1021.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 13:45:00 | 1028.30 | 1027.09 | 1021.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 1012.10 | 1024.12 | 1021.07 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 12:15:00 | 1008.00 | 1017.95 | 1018.86 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 1032.40 | 1019.81 | 1018.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 1042.40 | 1024.32 | 1020.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 14:15:00 | 1078.30 | 1082.50 | 1068.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 14:30:00 | 1078.95 | 1082.50 | 1068.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 1081.20 | 1082.24 | 1069.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 1099.30 | 1082.24 | 1069.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-10 10:15:00 | 1209.23 | 1126.46 | 1106.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 11:15:00 | 1263.20 | 1273.05 | 1273.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 12:15:00 | 1256.25 | 1269.69 | 1272.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 10:15:00 | 1242.10 | 1236.47 | 1246.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 11:00:00 | 1242.10 | 1236.47 | 1246.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1233.45 | 1232.02 | 1239.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 11:15:00 | 1226.05 | 1232.05 | 1238.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 15:15:00 | 1226.00 | 1228.59 | 1234.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 1257.55 | 1233.97 | 1236.07 | SL hit (close>static) qty=1.00 sl=1245.95 alert=retest2 |

### Cycle 30 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 1248.85 | 1239.43 | 1238.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 1250.50 | 1241.65 | 1239.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 1252.05 | 1253.83 | 1248.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 15:00:00 | 1252.05 | 1253.83 | 1248.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1259.30 | 1254.31 | 1249.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 1250.40 | 1254.31 | 1249.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 1258.10 | 1256.90 | 1252.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:45:00 | 1254.30 | 1256.90 | 1252.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 15:15:00 | 1250.00 | 1255.52 | 1252.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 1279.55 | 1255.52 | 1252.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 13:00:00 | 1263.55 | 1276.53 | 1271.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 13:15:00 | 1245.60 | 1270.34 | 1269.48 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 14:15:00 | 1258.70 | 1268.01 | 1268.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 09:15:00 | 1226.00 | 1247.38 | 1255.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 12:15:00 | 1221.25 | 1214.93 | 1227.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-07 12:45:00 | 1221.10 | 1214.93 | 1227.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 13:15:00 | 1221.50 | 1216.24 | 1227.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1213.10 | 1219.70 | 1225.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 1218.00 | 1202.92 | 1203.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 15:15:00 | 1218.00 | 1205.93 | 1204.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 15:15:00 | 1218.00 | 1205.93 | 1204.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 1236.00 | 1211.95 | 1207.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 15:15:00 | 1290.00 | 1293.33 | 1280.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:15:00 | 1307.95 | 1293.33 | 1280.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 1292.35 | 1298.13 | 1288.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 1295.10 | 1298.13 | 1288.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 1287.35 | 1295.97 | 1287.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 1287.35 | 1295.97 | 1287.99 | SL hit (close<ema400) qty=1.00 sl=1287.99 alert=retest1 |

### Cycle 33 — SELL (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 14:15:00 | 1353.05 | 1374.69 | 1377.02 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 1400.05 | 1381.67 | 1379.23 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 1350.00 | 1373.96 | 1376.64 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 12:15:00 | 1385.15 | 1377.09 | 1377.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 1399.80 | 1388.14 | 1382.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 09:15:00 | 1395.60 | 1406.25 | 1400.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 1395.60 | 1406.25 | 1400.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1395.60 | 1406.25 | 1400.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:00:00 | 1395.60 | 1406.25 | 1400.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 1392.00 | 1403.40 | 1399.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:00:00 | 1392.00 | 1403.40 | 1399.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 1393.75 | 1401.47 | 1398.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:30:00 | 1389.65 | 1401.47 | 1398.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 1404.95 | 1399.33 | 1398.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 10:15:00 | 1409.45 | 1400.89 | 1399.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 12:15:00 | 1392.00 | 1397.40 | 1397.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 1392.00 | 1397.40 | 1397.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 15:15:00 | 1380.20 | 1392.67 | 1395.51 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 1433.80 | 1400.90 | 1398.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 1471.00 | 1455.30 | 1445.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 1456.50 | 1465.07 | 1456.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1456.50 | 1465.07 | 1456.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1456.50 | 1465.07 | 1456.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 14:00:00 | 1484.30 | 1460.92 | 1456.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:30:00 | 1478.90 | 1468.47 | 1462.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 15:15:00 | 1456.90 | 1459.20 | 1459.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 15:15:00 | 1456.90 | 1459.20 | 1459.26 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 1478.65 | 1463.09 | 1461.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 11:15:00 | 1492.30 | 1469.29 | 1464.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 1461.05 | 1472.61 | 1468.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 1461.05 | 1472.61 | 1468.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1461.05 | 1472.61 | 1468.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 10:00:00 | 1461.05 | 1472.61 | 1468.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 1470.30 | 1472.15 | 1468.60 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 13:15:00 | 1457.65 | 1465.09 | 1465.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 1448.25 | 1461.72 | 1464.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 1466.85 | 1459.11 | 1462.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 1466.85 | 1459.11 | 1462.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1466.85 | 1459.11 | 1462.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:30:00 | 1462.30 | 1459.11 | 1462.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1461.60 | 1459.61 | 1462.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 11:30:00 | 1450.95 | 1456.04 | 1460.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 13:30:00 | 1448.25 | 1451.38 | 1457.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 14:15:00 | 1500.10 | 1461.12 | 1461.36 | SL hit (close>static) qty=1.00 sl=1468.30 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 15:15:00 | 1498.00 | 1468.50 | 1464.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 10:15:00 | 1528.35 | 1488.45 | 1474.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-24 14:15:00 | 1479.75 | 1489.21 | 1479.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 1479.75 | 1489.21 | 1479.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1479.75 | 1489.21 | 1479.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:30:00 | 1480.80 | 1489.21 | 1479.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 1479.00 | 1487.17 | 1479.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 09:15:00 | 1484.35 | 1487.17 | 1479.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 1461.10 | 1480.54 | 1477.99 | SL hit (close<static) qty=1.00 sl=1475.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 1446.45 | 1471.21 | 1474.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 15:15:00 | 1443.60 | 1459.17 | 1467.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 1460.85 | 1459.50 | 1466.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 1460.85 | 1459.50 | 1466.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1460.85 | 1459.50 | 1466.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:00:00 | 1460.85 | 1459.50 | 1466.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 1446.30 | 1428.87 | 1440.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:45:00 | 1447.60 | 1428.87 | 1440.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1445.00 | 1432.09 | 1440.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:15:00 | 1432.15 | 1432.09 | 1440.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 1432.15 | 1432.10 | 1439.74 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 1450.00 | 1443.48 | 1443.24 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 1426.15 | 1440.39 | 1441.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 11:15:00 | 1412.40 | 1434.79 | 1439.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 1433.15 | 1418.98 | 1428.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 1433.15 | 1418.98 | 1428.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1433.15 | 1418.98 | 1428.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:30:00 | 1431.45 | 1418.98 | 1428.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1436.80 | 1422.54 | 1428.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:30:00 | 1434.60 | 1422.54 | 1428.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 1422.15 | 1423.22 | 1428.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:15:00 | 1422.90 | 1423.22 | 1428.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 1414.90 | 1421.56 | 1426.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 14:45:00 | 1410.25 | 1419.63 | 1425.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 15:15:00 | 1399.00 | 1408.16 | 1415.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1435.10 | 1412.08 | 1415.59 | SL hit (close>static) qty=1.00 sl=1429.90 alert=retest2 |

### Cycle 46 — BUY (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 11:15:00 | 1430.00 | 1418.31 | 1417.97 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 15:15:00 | 1412.00 | 1418.21 | 1418.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 10:15:00 | 1399.50 | 1413.55 | 1416.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 15:15:00 | 1407.00 | 1404.96 | 1410.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-08 09:15:00 | 1413.80 | 1404.96 | 1410.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1417.05 | 1407.38 | 1410.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 1419.85 | 1407.38 | 1410.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 1399.70 | 1405.84 | 1409.74 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 12:15:00 | 1429.70 | 1414.04 | 1413.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 14:15:00 | 1439.00 | 1421.45 | 1416.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 1416.10 | 1423.32 | 1419.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 10:15:00 | 1416.10 | 1423.32 | 1419.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 1416.10 | 1423.32 | 1419.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 1416.10 | 1423.32 | 1419.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 1429.80 | 1424.62 | 1420.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:30:00 | 1417.95 | 1424.62 | 1420.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 12:15:00 | 1421.45 | 1423.98 | 1420.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 12:45:00 | 1419.80 | 1423.98 | 1420.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 1434.35 | 1426.06 | 1421.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 13:30:00 | 1412.20 | 1426.06 | 1421.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 1430.00 | 1426.85 | 1422.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:45:00 | 1422.75 | 1426.85 | 1422.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 1429.00 | 1427.28 | 1422.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 1462.00 | 1427.28 | 1422.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1450.00 | 1431.82 | 1425.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 11:00:00 | 1489.15 | 1443.29 | 1431.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 10:30:00 | 1487.00 | 1481.00 | 1460.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 12:15:00 | 1553.70 | 1562.85 | 1563.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 1553.70 | 1562.85 | 1563.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 1550.00 | 1560.28 | 1562.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 1472.65 | 1459.90 | 1475.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 1472.65 | 1459.90 | 1475.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1472.65 | 1459.90 | 1475.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 1431.45 | 1454.07 | 1465.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 10:00:00 | 1423.45 | 1447.95 | 1461.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 1510.00 | 1472.53 | 1467.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 1510.00 | 1472.53 | 1467.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 1516.85 | 1486.79 | 1475.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 1518.90 | 1520.81 | 1503.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 10:00:00 | 1518.90 | 1520.81 | 1503.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 1500.05 | 1512.88 | 1503.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 13:00:00 | 1500.05 | 1512.88 | 1503.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 1492.15 | 1508.74 | 1502.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:00:00 | 1492.15 | 1508.74 | 1502.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 1492.75 | 1505.54 | 1501.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 1491.00 | 1505.54 | 1501.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 1467.05 | 1496.00 | 1497.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 1442.30 | 1485.26 | 1492.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 1391.55 | 1390.72 | 1421.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:45:00 | 1393.45 | 1390.72 | 1421.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1397.00 | 1382.91 | 1401.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:30:00 | 1407.95 | 1382.91 | 1401.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 1295.00 | 1280.17 | 1296.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 1302.55 | 1280.17 | 1296.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1293.05 | 1282.75 | 1296.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:30:00 | 1279.15 | 1280.80 | 1294.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:00:00 | 1273.00 | 1280.80 | 1294.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 12:30:00 | 1277.50 | 1281.05 | 1291.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 10:00:00 | 1276.85 | 1283.03 | 1289.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 1288.90 | 1284.20 | 1289.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 11:00:00 | 1288.90 | 1284.20 | 1289.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 1277.40 | 1282.84 | 1288.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 11:30:00 | 1271.65 | 1280.73 | 1285.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 14:15:00 | 1269.95 | 1277.75 | 1282.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 09:45:00 | 1270.45 | 1274.02 | 1279.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 10:30:00 | 1271.05 | 1267.93 | 1272.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 1267.00 | 1266.43 | 1270.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 14:30:00 | 1269.85 | 1266.43 | 1270.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 1274.00 | 1268.52 | 1270.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-22 13:15:00 | 1274.95 | 1272.21 | 1271.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 1274.95 | 1272.21 | 1271.88 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 1267.00 | 1270.82 | 1271.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 10:15:00 | 1262.70 | 1268.41 | 1270.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 14:15:00 | 1259.25 | 1253.29 | 1258.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 1259.25 | 1253.29 | 1258.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 1259.25 | 1253.29 | 1258.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 1259.25 | 1253.29 | 1258.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1254.05 | 1253.44 | 1258.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 1268.10 | 1253.44 | 1258.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1262.40 | 1255.23 | 1258.43 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 10:15:00 | 1284.40 | 1261.07 | 1260.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 1318.45 | 1278.93 | 1269.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 09:15:00 | 1317.95 | 1350.45 | 1326.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 1317.95 | 1350.45 | 1326.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1317.95 | 1350.45 | 1326.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:45:00 | 1317.60 | 1350.45 | 1326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 1324.50 | 1345.26 | 1326.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:45:00 | 1313.70 | 1345.26 | 1326.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 1320.00 | 1340.21 | 1325.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 12:00:00 | 1320.00 | 1340.21 | 1325.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 12:15:00 | 1316.80 | 1335.53 | 1324.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 13:00:00 | 1316.80 | 1335.53 | 1324.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 13:15:00 | 1315.80 | 1331.58 | 1323.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 13:45:00 | 1315.70 | 1331.58 | 1323.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 10:15:00 | 1308.85 | 1320.18 | 1320.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 1302.50 | 1309.93 | 1314.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 10:15:00 | 1294.95 | 1292.01 | 1299.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-08 11:00:00 | 1294.95 | 1292.01 | 1299.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 11:15:00 | 1299.35 | 1293.48 | 1299.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 13:45:00 | 1291.60 | 1297.62 | 1299.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 14:45:00 | 1290.90 | 1296.85 | 1298.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 15:15:00 | 1290.50 | 1296.85 | 1298.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 10:45:00 | 1290.40 | 1293.17 | 1296.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 1300.00 | 1294.53 | 1296.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:00:00 | 1300.00 | 1294.53 | 1296.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 1304.10 | 1296.45 | 1297.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 1312.35 | 1299.63 | 1298.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 1312.35 | 1299.63 | 1298.79 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 1292.60 | 1300.48 | 1300.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 1280.70 | 1295.69 | 1298.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 11:15:00 | 1295.00 | 1293.84 | 1297.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 11:30:00 | 1294.95 | 1293.84 | 1297.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1311.60 | 1297.39 | 1298.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:00:00 | 1311.60 | 1297.39 | 1298.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 1300.05 | 1297.92 | 1298.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:30:00 | 1311.75 | 1297.92 | 1298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 1300.00 | 1298.98 | 1298.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 1306.20 | 1298.98 | 1298.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 1305.20 | 1300.22 | 1299.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 10:15:00 | 1314.25 | 1303.03 | 1300.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 1326.70 | 1326.97 | 1317.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-18 13:30:00 | 1324.20 | 1326.97 | 1317.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 1316.30 | 1324.84 | 1317.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:30:00 | 1316.50 | 1324.84 | 1317.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 1317.15 | 1323.30 | 1317.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 1301.60 | 1323.30 | 1317.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1305.85 | 1319.81 | 1316.26 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 11:15:00 | 1305.65 | 1313.78 | 1313.94 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 1322.15 | 1313.80 | 1313.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 10:15:00 | 1356.95 | 1339.68 | 1335.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 1341.55 | 1349.45 | 1343.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 09:15:00 | 1341.55 | 1349.45 | 1343.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1341.55 | 1349.45 | 1343.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 1341.55 | 1349.45 | 1343.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 1358.00 | 1351.16 | 1344.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:45:00 | 1340.40 | 1351.16 | 1344.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1353.00 | 1353.70 | 1348.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 1353.00 | 1353.70 | 1348.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 1360.00 | 1354.96 | 1349.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:30:00 | 1347.70 | 1353.48 | 1349.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 1342.30 | 1351.24 | 1348.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 1342.30 | 1351.24 | 1348.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 1347.00 | 1350.39 | 1348.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:30:00 | 1353.80 | 1349.93 | 1348.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:45:00 | 1359.65 | 1350.18 | 1348.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 10:30:00 | 1355.45 | 1351.07 | 1349.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:00:00 | 1354.35 | 1351.73 | 1349.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 1358.00 | 1354.08 | 1351.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:30:00 | 1349.75 | 1352.07 | 1350.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 1342.25 | 1350.10 | 1350.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 10:15:00 | 1342.25 | 1350.10 | 1350.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 11:15:00 | 1339.50 | 1347.98 | 1349.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 13:15:00 | 1309.30 | 1303.35 | 1315.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 13:30:00 | 1310.25 | 1303.35 | 1315.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 1305.00 | 1303.68 | 1314.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 14:45:00 | 1322.15 | 1303.68 | 1314.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 1300.00 | 1302.94 | 1313.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1290.40 | 1302.94 | 1313.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 13:00:00 | 1294.60 | 1295.65 | 1305.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:15:00 | 1287.65 | 1295.78 | 1304.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 15:15:00 | 1287.00 | 1297.20 | 1304.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1287.00 | 1295.16 | 1303.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:45:00 | 1283.10 | 1292.42 | 1296.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:30:00 | 1281.30 | 1281.21 | 1285.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 12:15:00 | 1299.50 | 1289.13 | 1288.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1299.50 | 1289.13 | 1288.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1310.00 | 1293.31 | 1290.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 11:15:00 | 1358.60 | 1361.38 | 1348.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 12:00:00 | 1358.60 | 1361.38 | 1348.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 1345.45 | 1356.91 | 1349.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:00:00 | 1345.45 | 1356.91 | 1349.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 1346.00 | 1354.73 | 1348.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:15:00 | 1334.20 | 1354.73 | 1348.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1334.20 | 1350.62 | 1347.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 1329.55 | 1350.62 | 1347.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 11:15:00 | 1317.95 | 1340.11 | 1342.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 1295.95 | 1327.53 | 1336.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 12:15:00 | 1299.90 | 1298.04 | 1310.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 13:00:00 | 1299.90 | 1298.04 | 1310.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1316.90 | 1300.61 | 1307.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 1321.05 | 1300.61 | 1307.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1317.00 | 1303.89 | 1308.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:00:00 | 1317.00 | 1303.89 | 1308.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1308.50 | 1308.16 | 1309.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 1313.15 | 1308.16 | 1309.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 1310.00 | 1308.53 | 1309.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 1316.00 | 1308.53 | 1309.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1314.15 | 1309.65 | 1309.77 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1313.90 | 1310.50 | 1310.14 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 1304.20 | 1310.17 | 1310.21 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 1312.55 | 1310.16 | 1310.04 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 1306.95 | 1309.52 | 1309.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1294.70 | 1306.16 | 1308.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 1299.80 | 1298.35 | 1302.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 13:00:00 | 1299.80 | 1298.35 | 1302.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1292.55 | 1294.15 | 1299.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 1294.85 | 1294.15 | 1299.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1300.15 | 1293.81 | 1296.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 1301.20 | 1293.81 | 1296.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1301.30 | 1295.31 | 1297.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 1296.60 | 1295.31 | 1297.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 13:15:00 | 1303.30 | 1297.91 | 1297.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 13:15:00 | 1303.30 | 1297.91 | 1297.68 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 1291.80 | 1296.69 | 1297.14 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 1302.00 | 1298.20 | 1297.74 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 13:15:00 | 1294.05 | 1297.35 | 1297.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 15:15:00 | 1289.90 | 1294.68 | 1296.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1296.55 | 1295.06 | 1296.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1296.55 | 1295.06 | 1296.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1296.55 | 1295.06 | 1296.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 1288.80 | 1295.06 | 1296.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 1282.40 | 1282.61 | 1289.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 13:15:00 | 1304.95 | 1282.84 | 1280.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 1304.95 | 1282.84 | 1280.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1313.00 | 1294.61 | 1287.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 1459.75 | 1461.10 | 1427.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:45:00 | 1458.25 | 1461.10 | 1427.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1428.65 | 1454.82 | 1443.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 1428.65 | 1454.82 | 1443.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1399.40 | 1443.73 | 1439.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:45:00 | 1395.00 | 1443.73 | 1439.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 1399.90 | 1434.97 | 1436.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 10:15:00 | 1388.00 | 1406.18 | 1418.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 12:15:00 | 1395.50 | 1386.74 | 1398.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:00:00 | 1395.50 | 1386.74 | 1398.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1397.00 | 1388.79 | 1398.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 1386.00 | 1392.40 | 1398.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 1400.10 | 1394.62 | 1397.09 | SL hit (close>static) qty=1.00 sl=1400.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 1418.50 | 1402.09 | 1400.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1428.10 | 1416.81 | 1409.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 1418.00 | 1419.29 | 1413.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 1405.70 | 1416.34 | 1413.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1405.70 | 1416.34 | 1413.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1404.00 | 1416.34 | 1413.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1405.35 | 1414.15 | 1412.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 1402.90 | 1414.15 | 1412.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1425.55 | 1416.24 | 1413.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:45:00 | 1420.95 | 1416.24 | 1413.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1410.25 | 1417.05 | 1415.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 1410.25 | 1417.05 | 1415.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1408.50 | 1415.34 | 1414.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 1407.95 | 1415.34 | 1414.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 1405.80 | 1413.06 | 1413.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 1399.00 | 1408.71 | 1411.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1399.95 | 1396.82 | 1402.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1399.95 | 1396.82 | 1402.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1399.95 | 1396.82 | 1402.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 1405.65 | 1396.82 | 1402.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1444.90 | 1406.44 | 1406.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 1444.90 | 1406.44 | 1406.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1451.30 | 1415.41 | 1410.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 1469.30 | 1426.19 | 1415.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 15:15:00 | 1463.00 | 1470.89 | 1453.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 09:15:00 | 1489.65 | 1470.89 | 1453.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1486.05 | 1473.92 | 1456.09 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 1448.00 | 1456.01 | 1456.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 12:15:00 | 1437.70 | 1450.32 | 1453.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 1432.00 | 1430.32 | 1438.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 1432.00 | 1430.32 | 1438.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1450.80 | 1435.16 | 1439.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:00:00 | 1450.80 | 1435.16 | 1439.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1455.65 | 1439.26 | 1440.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 1455.65 | 1439.26 | 1440.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 1453.20 | 1442.05 | 1441.96 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 1420.00 | 1438.69 | 1441.03 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 1486.00 | 1437.99 | 1436.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 1529.00 | 1456.19 | 1444.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 1511.95 | 1521.00 | 1504.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 1511.95 | 1521.00 | 1504.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1526.40 | 1522.24 | 1507.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 1572.10 | 1528.71 | 1520.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 1465.00 | 1515.82 | 1522.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1465.00 | 1515.82 | 1522.43 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1511.00 | 1495.47 | 1495.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 1524.00 | 1507.08 | 1501.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1502.05 | 1506.08 | 1501.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1502.05 | 1506.08 | 1501.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1502.05 | 1506.08 | 1501.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 1502.05 | 1506.08 | 1501.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 1509.40 | 1506.74 | 1502.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1530.00 | 1508.39 | 1504.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 12:15:00 | 1546.60 | 1579.87 | 1582.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 1546.60 | 1579.87 | 1582.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 1534.95 | 1567.86 | 1576.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1581.90 | 1567.49 | 1574.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1581.90 | 1567.49 | 1574.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1581.90 | 1567.49 | 1574.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 1547.50 | 1563.72 | 1570.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 1595.65 | 1569.96 | 1570.64 | SL hit (close>static) qty=1.00 sl=1590.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 1623.00 | 1580.57 | 1575.40 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 1559.55 | 1571.70 | 1572.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 1543.05 | 1564.34 | 1568.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 09:15:00 | 1547.35 | 1511.82 | 1523.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1547.35 | 1511.82 | 1523.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1547.35 | 1511.82 | 1523.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:30:00 | 1569.20 | 1511.82 | 1523.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1558.75 | 1521.21 | 1527.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 1558.75 | 1521.21 | 1527.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 1551.05 | 1532.88 | 1531.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 1560.00 | 1544.66 | 1537.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 1536.70 | 1560.82 | 1553.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 1536.70 | 1560.82 | 1553.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1536.70 | 1560.82 | 1553.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1539.20 | 1560.82 | 1553.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1555.80 | 1559.82 | 1553.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 1559.00 | 1558.37 | 1553.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 1559.05 | 1558.51 | 1554.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 15:00:00 | 1560.00 | 1558.80 | 1554.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-19 09:15:00 | 1714.90 | 1612.12 | 1585.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1815.05 | 1865.89 | 1866.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1799.10 | 1852.53 | 1860.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1824.35 | 1822.27 | 1838.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:00:00 | 1824.35 | 1822.27 | 1838.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1875.00 | 1831.00 | 1838.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 1875.00 | 1831.00 | 1838.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 1902.15 | 1845.23 | 1844.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 1940.00 | 1901.61 | 1881.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 15:15:00 | 1917.60 | 1920.64 | 1900.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 09:15:00 | 1928.50 | 1920.64 | 1900.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1969.85 | 1930.48 | 1906.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 2061.65 | 1962.22 | 1936.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 2005.00 | 2034.26 | 2031.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 12:15:00 | 2014.35 | 2027.22 | 2028.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 2014.35 | 2027.22 | 2028.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 12:15:00 | 1999.25 | 2016.19 | 2021.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 14:15:00 | 2018.05 | 2014.15 | 2019.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 2018.05 | 2014.15 | 2019.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 2018.05 | 2014.15 | 2019.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 2018.05 | 2014.15 | 2019.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1997.95 | 2010.89 | 2017.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 1995.00 | 2008.32 | 2015.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:30:00 | 1992.30 | 2004.82 | 2013.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 1992.40 | 1995.41 | 2000.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:15:00 | 1994.30 | 1995.41 | 2000.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1952.80 | 1940.44 | 1956.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 1939.70 | 1946.88 | 1955.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 10:00:00 | 1934.60 | 1946.88 | 1955.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 1941.05 | 1943.09 | 1951.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 13:15:00 | 1972.95 | 1947.61 | 1952.22 | SL hit (close>static) qty=1.00 sl=1969.45 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 2010.00 | 1960.09 | 1957.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 15:15:00 | 2017.95 | 1971.66 | 1962.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 11:15:00 | 1973.00 | 1974.32 | 1966.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 12:00:00 | 1973.00 | 1974.32 | 1966.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1958.70 | 1971.20 | 1965.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 1958.70 | 1971.20 | 1965.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1955.00 | 1967.96 | 1964.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 1958.55 | 1967.96 | 1964.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1955.00 | 1965.37 | 1964.00 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 1941.15 | 1960.62 | 1962.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1925.75 | 1953.65 | 1958.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 1935.00 | 1932.75 | 1944.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 1937.50 | 1932.75 | 1944.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1947.00 | 1935.60 | 1944.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 1948.60 | 1935.60 | 1944.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1941.90 | 1936.86 | 1944.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 1930.50 | 1939.45 | 1943.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:30:00 | 1923.00 | 1934.65 | 1940.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 13:00:00 | 1930.00 | 1912.65 | 1922.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:15:00 | 1925.00 | 1922.36 | 1925.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1913.30 | 1920.97 | 1924.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 1960.00 | 1928.77 | 1927.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 1960.00 | 1928.77 | 1927.38 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 14:15:00 | 1926.30 | 1935.48 | 1935.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 1919.00 | 1932.18 | 1934.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 12:15:00 | 1914.00 | 1912.67 | 1919.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:00:00 | 1914.00 | 1912.67 | 1919.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1979.50 | 1918.94 | 1919.41 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 1985.05 | 1932.16 | 1925.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 2019.95 | 1949.72 | 1933.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 2002.30 | 2010.62 | 1977.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 10:45:00 | 2004.85 | 2010.62 | 1977.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1976.85 | 1999.78 | 1986.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1958.85 | 1999.78 | 1986.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1981.95 | 1996.21 | 1986.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 1982.85 | 1996.21 | 1986.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1973.70 | 1990.40 | 1985.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 1973.70 | 1990.40 | 1985.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1974.30 | 1987.18 | 1984.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 1968.10 | 1987.18 | 1984.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 1970.40 | 1981.38 | 1981.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1932.30 | 1971.56 | 1977.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 1908.85 | 1902.22 | 1929.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 14:15:00 | 1937.25 | 1911.56 | 1928.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1937.25 | 1911.56 | 1928.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1937.25 | 1911.56 | 1928.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1929.80 | 1915.21 | 1928.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1940.40 | 1915.21 | 1928.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1931.95 | 1918.56 | 1929.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:30:00 | 1922.70 | 1920.35 | 1929.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 1924.20 | 1921.04 | 1928.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 1919.95 | 1906.93 | 1906.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1919.95 | 1906.93 | 1906.80 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 1901.00 | 1906.03 | 1906.55 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 1912.35 | 1907.29 | 1907.08 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 1906.90 | 1910.61 | 1911.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1896.90 | 1907.12 | 1909.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 1920.40 | 1897.21 | 1901.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1920.40 | 1897.21 | 1901.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1920.40 | 1897.21 | 1901.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 1920.40 | 1897.21 | 1901.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1894.15 | 1896.60 | 1900.93 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1915.00 | 1904.65 | 1904.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 1929.35 | 1909.59 | 1906.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 1909.20 | 1912.45 | 1908.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 1909.20 | 1912.45 | 1908.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1909.20 | 1912.45 | 1908.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 1906.60 | 1912.45 | 1908.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1899.20 | 1909.80 | 1907.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 1899.20 | 1909.80 | 1907.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1890.80 | 1906.00 | 1906.32 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 09:15:00 | 1923.50 | 1906.80 | 1906.12 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 1889.00 | 1904.30 | 1905.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 1882.05 | 1899.85 | 1903.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 13:15:00 | 1846.15 | 1843.97 | 1859.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 13:45:00 | 1851.10 | 1843.97 | 1859.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1868.65 | 1848.91 | 1860.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 1868.65 | 1848.91 | 1860.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1855.55 | 1850.23 | 1859.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 1836.75 | 1850.23 | 1859.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 1870.50 | 1823.54 | 1822.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1870.50 | 1823.54 | 1822.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1909.55 | 1857.41 | 1840.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1958.55 | 1996.42 | 1960.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1958.55 | 1996.42 | 1960.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1958.55 | 1996.42 | 1960.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1958.55 | 1996.42 | 1960.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1968.95 | 1990.92 | 1961.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:00:00 | 1991.70 | 1991.08 | 1964.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:30:00 | 1994.45 | 1989.41 | 1965.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 2014.70 | 1974.83 | 1969.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 2009.60 | 2044.79 | 2048.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 2009.60 | 2044.79 | 2048.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1962.95 | 2010.14 | 2022.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 1990.90 | 1970.86 | 1987.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 12:15:00 | 1990.90 | 1970.86 | 1987.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1990.90 | 1970.86 | 1987.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:45:00 | 1980.95 | 1970.86 | 1987.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1965.80 | 1969.85 | 1985.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 1957.15 | 1965.81 | 1982.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 2005.05 | 1971.29 | 1981.84 | SL hit (close>static) qty=1.00 sl=1993.10 alert=retest2 |

### Cycle 106 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 1999.30 | 1988.46 | 1987.77 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 11:15:00 | 1972.95 | 1986.25 | 1987.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 12:15:00 | 1966.80 | 1982.36 | 1985.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1962.25 | 1953.56 | 1962.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1962.25 | 1953.56 | 1962.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1962.25 | 1953.56 | 1962.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 1962.25 | 1953.56 | 1962.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1976.00 | 1958.05 | 1964.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 1976.00 | 1958.05 | 1964.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1970.30 | 1960.50 | 1964.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 1978.30 | 1960.50 | 1964.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1964.15 | 1961.23 | 1964.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 14:45:00 | 1957.30 | 1961.86 | 1964.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1957.95 | 1962.49 | 1964.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 13:15:00 | 1970.75 | 1965.15 | 1964.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 1970.75 | 1965.15 | 1964.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 2028.95 | 1980.96 | 1972.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 12:15:00 | 2346.30 | 2364.09 | 2324.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 12:15:00 | 2346.30 | 2364.09 | 2324.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 2346.30 | 2364.09 | 2324.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:45:00 | 2319.60 | 2364.09 | 2324.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 2353.35 | 2380.05 | 2355.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:45:00 | 2353.50 | 2380.05 | 2355.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 2352.35 | 2374.51 | 2355.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 2352.35 | 2374.51 | 2355.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 2359.00 | 2371.41 | 2355.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 2403.85 | 2371.41 | 2355.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 2341.60 | 2365.44 | 2354.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 2341.60 | 2365.44 | 2354.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 2363.20 | 2365.00 | 2355.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 2369.30 | 2365.00 | 2355.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 2375.55 | 2367.30 | 2359.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:00:00 | 2369.50 | 2366.12 | 2360.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 2385.85 | 2365.71 | 2361.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 2407.00 | 2433.81 | 2424.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 2412.10 | 2433.81 | 2424.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 2392.40 | 2425.53 | 2421.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 2371.10 | 2425.53 | 2421.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 2391.90 | 2418.80 | 2419.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 2391.90 | 2418.80 | 2419.08 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 2433.40 | 2420.07 | 2418.82 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 2407.40 | 2417.54 | 2417.78 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 2428.00 | 2419.56 | 2418.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 2505.00 | 2439.86 | 2428.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 2443.40 | 2459.11 | 2444.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 14:15:00 | 2443.40 | 2459.11 | 2444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 2443.40 | 2459.11 | 2444.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 2443.40 | 2459.11 | 2444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 2440.00 | 2455.29 | 2444.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 2418.70 | 2455.29 | 2444.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 2378.10 | 2439.85 | 2438.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 2378.10 | 2439.85 | 2438.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 2399.75 | 2431.83 | 2434.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 2360.95 | 2398.92 | 2415.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 2408.80 | 2396.59 | 2406.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 15:15:00 | 2408.80 | 2396.59 | 2406.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2408.80 | 2396.59 | 2406.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 2385.40 | 2392.69 | 2403.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 2363.10 | 2387.30 | 2399.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 2373.05 | 2365.37 | 2378.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:45:00 | 2386.50 | 2374.74 | 2380.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 15:15:00 | 2432.00 | 2392.33 | 2387.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 15:15:00 | 2432.00 | 2392.33 | 2387.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 2465.25 | 2409.25 | 2400.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 11:15:00 | 2480.00 | 2497.32 | 2459.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 12:00:00 | 2480.00 | 2497.32 | 2459.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2532.10 | 2499.47 | 2469.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 2576.95 | 2522.08 | 2502.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:30:00 | 2557.00 | 2550.11 | 2524.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 12:45:00 | 2555.95 | 2563.77 | 2542.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 2519.90 | 2535.67 | 2535.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 2519.90 | 2535.67 | 2535.74 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 2569.00 | 2534.48 | 2534.20 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 2507.95 | 2545.32 | 2545.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 2479.80 | 2532.22 | 2539.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 2323.25 | 2308.35 | 2365.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 2323.25 | 2308.35 | 2365.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 2340.00 | 2313.55 | 2345.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 2307.00 | 2313.55 | 2345.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 12:15:00 | 2191.65 | 2230.43 | 2254.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 2214.50 | 2208.16 | 2227.56 | SL hit (close>ema200) qty=0.50 sl=2208.16 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 2232.95 | 2198.09 | 2195.40 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 2141.55 | 2194.19 | 2195.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1963.00 | 2124.43 | 2158.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1903.40 | 1902.33 | 1989.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 1903.40 | 1902.33 | 1989.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1971.40 | 1933.69 | 1959.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 1971.40 | 1933.69 | 1959.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 2002.65 | 1947.48 | 1963.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 1994.20 | 1947.48 | 1963.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 2027.80 | 1983.95 | 1978.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 2034.80 | 2000.72 | 1987.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 2044.85 | 2064.31 | 2043.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 2044.85 | 2064.31 | 2043.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2044.85 | 2064.31 | 2043.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 2044.85 | 2064.31 | 2043.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 2068.05 | 2065.06 | 2045.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 2099.65 | 2071.98 | 2050.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 2028.60 | 2066.83 | 2052.28 | SL hit (close<static) qty=1.00 sl=2036.95 alert=retest2 |

### Cycle 121 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 2032.95 | 2044.05 | 2044.46 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 2108.05 | 2052.81 | 2047.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 10:15:00 | 2130.00 | 2068.25 | 2055.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 2176.55 | 2179.82 | 2144.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 2176.55 | 2179.82 | 2144.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2160.10 | 2172.24 | 2155.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2160.10 | 2172.24 | 2155.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2193.65 | 2176.52 | 2159.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 2197.65 | 2176.52 | 2159.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 2197.00 | 2193.33 | 2173.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 2136.80 | 2184.69 | 2173.08 | SL hit (close<static) qty=1.00 sl=2146.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 2080.00 | 2152.50 | 2159.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 2063.70 | 2113.78 | 2135.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 2086.55 | 2060.51 | 2088.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 2086.55 | 2060.51 | 2088.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 2086.55 | 2060.51 | 2088.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 2070.00 | 2060.51 | 2088.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 2081.10 | 2064.63 | 2088.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:15:00 | 2089.20 | 2064.63 | 2088.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 2091.85 | 2070.07 | 2088.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 2091.85 | 2070.07 | 2088.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 2036.00 | 2063.26 | 2083.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 2035.75 | 2063.38 | 2080.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:45:00 | 2030.25 | 2053.59 | 2072.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:45:00 | 2027.05 | 2047.34 | 2066.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 2023.75 | 2047.34 | 2066.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 1933.96 | 2000.33 | 2034.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1928.74 | 1972.67 | 2015.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1925.70 | 1972.67 | 2015.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1922.56 | 1972.67 | 2015.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 1950.20 | 1946.05 | 1986.93 | SL hit (close>ema200) qty=0.50 sl=1946.05 alert=retest2 |

### Cycle 124 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 2004.85 | 1965.19 | 1964.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 2022.00 | 1976.55 | 1970.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 14:15:00 | 2031.75 | 2049.49 | 2031.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 14:15:00 | 2031.75 | 2049.49 | 2031.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 2031.75 | 2049.49 | 2031.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 2031.35 | 2049.49 | 2031.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 2027.00 | 2044.99 | 2031.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 2027.65 | 2044.99 | 2031.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2009.85 | 2037.96 | 2029.09 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 1991.30 | 2020.01 | 2021.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 1968.75 | 2003.41 | 2013.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 1792.35 | 1756.15 | 1808.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 10:00:00 | 1792.35 | 1756.15 | 1808.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1784.05 | 1761.73 | 1805.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 1782.00 | 1761.73 | 1805.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 1791.05 | 1767.59 | 1804.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:45:00 | 1796.65 | 1767.59 | 1804.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1795.00 | 1773.07 | 1803.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 1792.95 | 1773.07 | 1803.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1809.80 | 1780.42 | 1804.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 1809.80 | 1780.42 | 1804.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1813.00 | 1786.94 | 1805.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:45:00 | 1809.00 | 1786.94 | 1805.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 1809.30 | 1791.41 | 1805.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 1851.35 | 1791.41 | 1805.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1831.95 | 1799.52 | 1807.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 1858.55 | 1799.52 | 1807.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1839.20 | 1807.45 | 1810.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 1843.90 | 1807.45 | 1810.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 1845.00 | 1814.96 | 1813.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1936.45 | 1851.35 | 1832.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 1940.00 | 1940.43 | 1911.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:15:00 | 1936.00 | 1940.43 | 1911.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1911.50 | 1929.14 | 1918.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1904.35 | 1929.14 | 1918.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1897.40 | 1922.79 | 1916.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1877.20 | 1922.79 | 1916.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 1905.00 | 1911.13 | 1911.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1886.40 | 1905.70 | 1908.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1876.35 | 1873.53 | 1887.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:00:00 | 1876.35 | 1873.53 | 1887.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1864.20 | 1871.67 | 1885.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 1876.25 | 1871.67 | 1885.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1871.60 | 1858.40 | 1871.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 1890.80 | 1858.40 | 1871.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1868.50 | 1860.42 | 1871.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:30:00 | 1859.00 | 1860.10 | 1869.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1891.20 | 1861.89 | 1866.68 | SL hit (close>static) qty=1.00 sl=1877.35 alert=retest2 |

### Cycle 128 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1916.30 | 1872.77 | 1871.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 11:15:00 | 1927.20 | 1908.86 | 1894.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 15:15:00 | 2041.50 | 2044.80 | 2010.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:15:00 | 2041.05 | 2044.80 | 2010.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2017.90 | 2042.65 | 2028.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 2024.55 | 2042.65 | 2028.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2007.45 | 2035.61 | 2026.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 2007.45 | 2035.61 | 2026.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1992.95 | 2021.35 | 2021.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1987.90 | 2006.29 | 2013.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 2012.35 | 1996.45 | 2003.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 2012.35 | 1996.45 | 2003.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 2012.35 | 1996.45 | 2003.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 2012.35 | 1996.45 | 2003.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2007.00 | 1998.56 | 2003.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2005.15 | 1998.56 | 2003.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 2022.00 | 2007.23 | 2007.14 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 2000.05 | 2005.80 | 2006.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 1990.00 | 2000.25 | 2003.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 1971.55 | 1971.27 | 1982.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:30:00 | 1964.15 | 1971.27 | 1982.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1976.60 | 1971.28 | 1979.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1976.60 | 1971.28 | 1979.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1977.00 | 1972.42 | 1979.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 2084.70 | 1972.42 | 1979.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 2036.90 | 1985.32 | 1984.61 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1953.00 | 1989.47 | 1993.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1945.90 | 1980.76 | 1989.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1869.80 | 1863.22 | 1907.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1882.60 | 1863.22 | 1907.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1863.00 | 1863.17 | 1903.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1780.40 | 1868.97 | 1889.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:45:00 | 1835.35 | 1800.88 | 1817.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 15:15:00 | 1869.00 | 1829.08 | 1827.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 1869.00 | 1829.08 | 1827.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1886.60 | 1840.59 | 1832.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1937.10 | 1947.11 | 1915.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1937.10 | 1947.11 | 1915.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1937.10 | 1947.11 | 1915.33 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 15:15:00 | 1908.60 | 1922.98 | 1924.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 11:15:00 | 1890.80 | 1909.14 | 1917.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 11:15:00 | 1894.00 | 1893.41 | 1903.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 11:15:00 | 1894.00 | 1893.41 | 1903.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1894.00 | 1893.41 | 1903.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 1900.50 | 1893.41 | 1903.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1902.30 | 1895.19 | 1903.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 1902.30 | 1895.19 | 1903.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 1901.90 | 1896.53 | 1903.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 1897.90 | 1896.53 | 1903.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 1887.20 | 1894.66 | 1901.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 1882.00 | 1892.73 | 1900.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 1903.10 | 1894.80 | 1900.40 | SL hit (close>static) qty=1.00 sl=1902.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 1911.00 | 1868.18 | 1868.14 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 1873.30 | 1881.17 | 1881.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1859.50 | 1876.84 | 1879.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 1876.40 | 1875.74 | 1878.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 13:15:00 | 1876.40 | 1875.74 | 1878.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1876.40 | 1875.74 | 1878.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 1876.40 | 1875.74 | 1878.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1887.60 | 1878.11 | 1879.59 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1891.00 | 1881.79 | 1881.07 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 11:15:00 | 1874.00 | 1880.55 | 1880.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 12:15:00 | 1866.70 | 1877.78 | 1879.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 13:15:00 | 1881.50 | 1878.53 | 1879.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 13:15:00 | 1881.50 | 1878.53 | 1879.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1881.50 | 1878.53 | 1879.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 1881.50 | 1878.53 | 1879.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 1908.60 | 1884.54 | 1882.22 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1861.00 | 1877.93 | 1879.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 1844.00 | 1865.21 | 1872.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1869.70 | 1855.81 | 1863.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 1869.70 | 1855.81 | 1863.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1869.70 | 1855.81 | 1863.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1869.70 | 1855.81 | 1863.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1877.20 | 1860.08 | 1864.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1880.80 | 1860.08 | 1864.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1907.80 | 1869.63 | 1868.39 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1857.10 | 1874.09 | 1875.60 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1931.60 | 1880.16 | 1876.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1938.50 | 1900.66 | 1887.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 2007.70 | 2009.81 | 1986.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 2007.70 | 2009.81 | 1986.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1994.00 | 2002.90 | 1989.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 1977.20 | 2002.90 | 1989.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1978.00 | 1997.39 | 1990.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 1978.00 | 1997.39 | 1990.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1985.40 | 1994.99 | 1990.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 1988.00 | 1994.99 | 1990.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1985.60 | 1993.11 | 1989.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:45:00 | 1991.40 | 1993.27 | 1990.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-20 09:15:00 | 2190.54 | 2128.57 | 2077.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 2188.10 | 2211.92 | 2214.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 2161.30 | 2175.68 | 2186.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 2173.00 | 2164.11 | 2174.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2173.00 | 2164.11 | 2174.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2173.00 | 2164.11 | 2174.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 2175.30 | 2164.11 | 2174.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 2172.00 | 2165.69 | 2174.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 2166.10 | 2165.69 | 2174.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 2160.10 | 2162.13 | 2168.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 2165.00 | 2163.59 | 2168.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:00:00 | 2164.50 | 2163.77 | 2167.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 2168.60 | 2164.74 | 2167.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:45:00 | 2167.90 | 2164.74 | 2167.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 2153.10 | 2162.41 | 2166.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 2114.40 | 2159.19 | 2164.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 2147.60 | 2155.28 | 2161.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:00:00 | 2146.30 | 2152.88 | 2158.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 2184.00 | 2161.17 | 2161.37 | SL hit (close>static) qty=1.00 sl=2170.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 2179.70 | 2164.88 | 2163.04 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 2160.60 | 2162.51 | 2162.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 2152.00 | 2160.41 | 2161.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 15:15:00 | 2037.00 | 2036.79 | 2060.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 2045.00 | 2036.79 | 2060.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2057.40 | 2040.91 | 2060.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 2054.40 | 2040.91 | 2060.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 2100.10 | 2052.75 | 2063.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 2087.00 | 2052.75 | 2063.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 2089.70 | 2060.14 | 2066.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:45:00 | 2088.70 | 2060.14 | 2066.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 2101.00 | 2075.44 | 2072.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 2122.20 | 2090.96 | 2080.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 2118.10 | 2126.16 | 2105.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 2118.10 | 2126.16 | 2105.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2110.30 | 2122.99 | 2105.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 2110.30 | 2122.99 | 2105.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2108.00 | 2119.99 | 2105.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2089.90 | 2119.99 | 2105.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2098.90 | 2115.77 | 2105.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 2112.50 | 2114.54 | 2105.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 2120.00 | 2114.65 | 2106.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 2093.20 | 2108.50 | 2108.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 12:15:00 | 2093.20 | 2108.50 | 2108.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 2082.40 | 2096.65 | 2102.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 2099.70 | 2091.47 | 2097.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 2099.70 | 2091.47 | 2097.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2099.70 | 2091.47 | 2097.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 2099.70 | 2091.47 | 2097.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 2085.90 | 2090.36 | 2096.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 2079.30 | 2085.90 | 2091.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 2063.60 | 2038.55 | 2046.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2063.80 | 2051.79 | 2051.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 2063.80 | 2051.79 | 2051.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 2102.80 | 2063.84 | 2056.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 2094.90 | 2094.98 | 2080.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 2094.90 | 2094.98 | 2080.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2080.70 | 2094.30 | 2085.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 2078.90 | 2094.30 | 2085.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2087.00 | 2092.84 | 2085.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2089.00 | 2092.84 | 2085.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2140.00 | 2102.27 | 2090.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 2164.00 | 2114.04 | 2103.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 2081.60 | 2105.55 | 2106.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 2081.60 | 2105.55 | 2106.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 1999.90 | 2071.82 | 2089.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2005.00 | 1996.89 | 2027.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 2007.10 | 2001.69 | 2015.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2007.10 | 2001.69 | 2015.54 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 2053.40 | 2023.22 | 2019.24 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 2013.50 | 2019.12 | 2019.88 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 2030.30 | 2021.35 | 2020.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2043.00 | 2029.07 | 2025.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 2024.20 | 2030.68 | 2027.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 2024.20 | 2030.68 | 2027.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2024.20 | 2030.68 | 2027.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 2024.20 | 2030.68 | 2027.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 2017.30 | 2028.01 | 2026.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 2017.30 | 2028.01 | 2026.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2023.50 | 2027.11 | 2026.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 2033.90 | 2027.11 | 2026.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 2016.00 | 2024.90 | 2025.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 2016.00 | 2024.90 | 2025.52 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 2044.00 | 2028.00 | 2026.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 2083.60 | 2047.53 | 2037.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 2070.60 | 2070.91 | 2057.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:30:00 | 2067.90 | 2070.91 | 2057.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2095.50 | 2100.73 | 2086.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2090.80 | 2100.73 | 2086.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2087.00 | 2100.42 | 2093.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 2090.90 | 2100.42 | 2093.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 2101.10 | 2100.56 | 2094.02 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 2087.80 | 2093.26 | 2093.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 2077.20 | 2087.93 | 2091.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2102.90 | 2088.75 | 2090.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2102.90 | 2088.75 | 2090.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2102.90 | 2088.75 | 2090.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 2082.50 | 2088.75 | 2090.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2100.00 | 2091.00 | 2091.42 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 2111.80 | 2095.16 | 2093.28 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 2080.00 | 2091.68 | 2093.18 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 2101.90 | 2094.06 | 2093.89 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 2091.00 | 2093.45 | 2093.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2066.30 | 2086.97 | 2090.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2074.90 | 2062.10 | 2074.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2074.90 | 2062.10 | 2074.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2074.90 | 2062.10 | 2074.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 2071.10 | 2062.10 | 2074.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2072.00 | 2064.08 | 2073.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 2070.00 | 2064.08 | 2073.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 2070.70 | 2066.30 | 2074.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 2059.00 | 2065.32 | 2072.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 2082.80 | 2067.01 | 2067.53 | SL hit (close>static) qty=1.00 sl=2081.10 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 2088.00 | 2071.20 | 2069.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2106.50 | 2078.26 | 2072.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2095.70 | 2118.08 | 2100.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2095.70 | 2118.08 | 2100.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2095.70 | 2118.08 | 2100.76 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 2070.90 | 2092.13 | 2093.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 2039.60 | 2081.63 | 2088.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 2012.00 | 2010.40 | 2029.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 2017.20 | 2010.40 | 2029.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 1971.10 | 1928.44 | 1949.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 1971.10 | 1928.44 | 1949.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1998.40 | 1942.43 | 1953.53 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 2069.00 | 1967.75 | 1964.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 2158.20 | 2043.54 | 2004.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 2101.10 | 2102.17 | 2075.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:45:00 | 2102.30 | 2102.17 | 2075.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2114.60 | 2107.77 | 2090.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 2130.30 | 2112.28 | 2094.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 2128.30 | 2132.27 | 2125.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 2123.10 | 2127.41 | 2124.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:00:00 | 2122.50 | 2135.38 | 2133.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 2120.80 | 2132.47 | 2132.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 2120.80 | 2132.47 | 2132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 2107.40 | 2122.13 | 2126.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 2156.10 | 2128.92 | 2129.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 2156.10 | 2128.92 | 2129.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 2156.10 | 2128.92 | 2129.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 2163.20 | 2128.92 | 2129.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 2164.10 | 2135.96 | 2132.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 12:15:00 | 2171.00 | 2142.97 | 2136.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 2173.60 | 2185.21 | 2169.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 2173.60 | 2185.21 | 2169.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 2173.60 | 2185.21 | 2169.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 2173.60 | 2185.21 | 2169.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 2179.00 | 2183.97 | 2170.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 2205.00 | 2183.97 | 2170.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 2190.20 | 2185.99 | 2177.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 2165.20 | 2181.83 | 2176.06 | SL hit (close<static) qty=1.00 sl=2166.20 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2125.00 | 2168.09 | 2170.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 2100.40 | 2137.28 | 2154.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 2119.80 | 2114.72 | 2136.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 2119.80 | 2114.72 | 2136.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 2143.40 | 2120.45 | 2137.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 2139.00 | 2120.45 | 2137.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2148.50 | 2126.06 | 2138.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 2148.50 | 2126.06 | 2138.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 2145.20 | 2129.89 | 2138.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 2128.50 | 2138.60 | 2141.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 2125.80 | 2119.30 | 2119.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 2125.80 | 2119.30 | 2119.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2154.60 | 2126.36 | 2122.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 2141.00 | 2151.75 | 2140.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 2141.00 | 2151.75 | 2140.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2141.00 | 2151.75 | 2140.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 2132.10 | 2151.75 | 2140.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2126.70 | 2146.74 | 2139.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2126.70 | 2146.74 | 2139.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2130.00 | 2143.39 | 2138.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 2137.50 | 2140.83 | 2137.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 2116.30 | 2132.75 | 2134.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 2116.30 | 2132.75 | 2134.52 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 2180.50 | 2139.46 | 2137.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 2192.20 | 2163.83 | 2152.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 2226.20 | 2231.43 | 2217.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 2259.00 | 2231.43 | 2217.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2251.30 | 2235.40 | 2220.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 2268.30 | 2239.84 | 2223.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 2278.10 | 2242.81 | 2231.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 2268.40 | 2249.74 | 2248.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2286.00 | 2313.15 | 2315.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 2286.00 | 2313.15 | 2315.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 2278.90 | 2306.30 | 2312.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 2012.00 | 1998.59 | 2039.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 2022.90 | 1998.59 | 2039.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 2049.60 | 2020.14 | 2037.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 2049.60 | 2020.14 | 2037.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2051.70 | 2026.45 | 2038.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 2051.70 | 2026.45 | 2038.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 2066.90 | 2034.54 | 2041.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 2045.60 | 2034.54 | 2041.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:15:00 | 2044.60 | 2037.91 | 2042.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 2035.70 | 2018.31 | 2016.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 2035.70 | 2018.31 | 2016.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 2046.90 | 2031.02 | 2024.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 2019.50 | 2042.33 | 2034.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2019.50 | 2042.33 | 2034.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2019.50 | 2042.33 | 2034.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 2019.50 | 2042.33 | 2034.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 2017.50 | 2037.37 | 2033.30 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 2012.20 | 2029.52 | 2030.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 2009.50 | 2025.52 | 2028.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1987.40 | 1987.07 | 2001.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 1987.40 | 1987.07 | 2001.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2000.50 | 1989.76 | 2001.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 2002.70 | 1989.76 | 2001.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 2005.00 | 1992.81 | 2001.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 2005.00 | 1992.81 | 2001.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1998.50 | 1993.94 | 2001.63 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 2024.00 | 2005.89 | 2005.61 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 1999.40 | 2004.59 | 2005.04 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 2018.90 | 2007.80 | 2006.31 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1996.30 | 2004.92 | 2005.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 1988.00 | 1999.16 | 2003.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 11:15:00 | 1992.10 | 1992.05 | 1998.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:45:00 | 1994.00 | 1992.05 | 1998.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 2020.00 | 1997.64 | 2000.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 2023.10 | 1997.64 | 2000.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 2016.10 | 2001.33 | 2001.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 2027.40 | 2001.33 | 2001.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2017.40 | 2004.54 | 2003.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 2033.00 | 2011.43 | 2006.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 2019.20 | 2023.84 | 2016.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 2019.20 | 2023.84 | 2016.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2024.90 | 2024.05 | 2017.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 2025.20 | 2022.44 | 2018.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 2015.00 | 2020.95 | 2018.03 | SL hit (close<static) qty=1.00 sl=2015.10 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 2003.70 | 2014.41 | 2015.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 15:15:00 | 1997.50 | 2011.02 | 2013.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 2013.80 | 2009.22 | 2012.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 2013.80 | 2009.22 | 2012.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 2013.80 | 2009.22 | 2012.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 2001.50 | 2008.30 | 2011.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 1997.40 | 2005.65 | 2008.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 2017.60 | 2010.48 | 2010.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 2017.60 | 2010.48 | 2010.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 2023.00 | 2015.30 | 2012.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2014.50 | 2017.25 | 2014.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2014.50 | 2017.25 | 2014.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2014.50 | 2017.25 | 2014.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 2032.00 | 2020.20 | 2015.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 2031.80 | 2025.00 | 2019.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:00:00 | 2036.90 | 2027.58 | 2023.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 2044.20 | 2030.03 | 2025.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 2025.70 | 2029.17 | 2025.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 2025.70 | 2029.17 | 2025.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 2009.00 | 2025.13 | 2024.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 2009.00 | 2025.13 | 2024.14 | SL hit (close<static) qty=1.00 sl=2012.10 alert=retest2 |

### Cycle 181 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 2001.70 | 2020.45 | 2022.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 1994.40 | 2007.88 | 2014.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 2029.50 | 2002.11 | 2006.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2029.50 | 2002.11 | 2006.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2029.50 | 2002.11 | 2006.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 2005.00 | 2002.69 | 2006.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 2010.00 | 2004.15 | 2006.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 13:15:00 | 2020.90 | 2010.37 | 2009.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 13:15:00 | 2020.90 | 2010.37 | 2009.11 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 1998.00 | 2007.90 | 2008.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1918.00 | 1989.89 | 1999.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 1947.90 | 1944.83 | 1963.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 1947.90 | 1944.83 | 1963.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1945.30 | 1938.48 | 1947.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 1945.30 | 1938.48 | 1947.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1942.00 | 1939.19 | 1946.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 1941.10 | 1941.78 | 1946.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 1936.20 | 1943.65 | 1946.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1962.00 | 1950.42 | 1948.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 1962.00 | 1950.42 | 1948.93 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1936.30 | 1947.99 | 1948.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 1930.10 | 1944.41 | 1946.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 1931.70 | 1929.58 | 1935.56 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 12:30:00 | 1925.10 | 1928.72 | 1934.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 13:00:00 | 1926.00 | 1928.72 | 1934.12 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1936.00 | 1930.18 | 1934.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 1936.00 | 1930.18 | 1934.30 | SL hit (close>ema400) qty=1.00 sl=1934.30 alert=retest1 |

### Cycle 186 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1920.00 | 1911.66 | 1910.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1923.30 | 1915.32 | 1912.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1906.30 | 1913.64 | 1912.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 1906.30 | 1913.64 | 1912.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1906.30 | 1913.64 | 1912.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1906.30 | 1913.64 | 1912.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1908.20 | 1912.55 | 1912.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1913.10 | 1912.55 | 1912.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1916.00 | 1912.10 | 1911.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1912.00 | 1914.59 | 1913.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1925.00 | 1913.80 | 1913.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 1910.90 | 1913.22 | 1913.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 1910.90 | 1913.22 | 1913.43 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1916.50 | 1913.87 | 1913.71 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 1910.30 | 1913.47 | 1913.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 1906.50 | 1911.32 | 1912.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1911.20 | 1909.88 | 1911.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1911.20 | 1909.88 | 1911.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1911.20 | 1909.88 | 1911.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1910.80 | 1909.88 | 1911.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1917.00 | 1911.31 | 1912.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 1917.00 | 1911.31 | 1912.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1912.00 | 1911.44 | 1912.04 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1918.60 | 1912.77 | 1912.54 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1898.20 | 1909.73 | 1911.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 1890.80 | 1902.39 | 1907.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1903.70 | 1898.42 | 1903.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1903.70 | 1898.42 | 1903.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1903.70 | 1898.42 | 1903.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 1904.00 | 1898.42 | 1903.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1910.00 | 1900.73 | 1903.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1908.90 | 1900.73 | 1903.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1916.00 | 1903.79 | 1905.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 1917.30 | 1903.79 | 1905.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1921.80 | 1907.39 | 1906.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1952.60 | 1916.43 | 1910.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1937.70 | 1938.81 | 1924.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1937.70 | 1938.81 | 1924.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1937.70 | 1938.81 | 1924.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1937.70 | 1938.81 | 1924.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1934.10 | 1935.86 | 1927.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 1945.50 | 1938.55 | 1929.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 1942.20 | 1936.47 | 1931.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1919.80 | 1933.14 | 1930.43 | SL hit (close<static) qty=1.00 sl=1927.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1915.80 | 1926.43 | 1927.65 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1974.50 | 1936.05 | 1931.91 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1885.70 | 1925.98 | 1927.71 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1952.30 | 1926.65 | 1924.85 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 1947.00 | 1950.96 | 1951.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1929.60 | 1945.67 | 1948.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1918.90 | 1913.88 | 1925.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1918.10 | 1913.88 | 1925.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1925.00 | 1914.75 | 1921.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1927.00 | 1914.75 | 1921.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1917.20 | 1915.24 | 1920.89 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1938.10 | 1925.15 | 1923.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 1950.00 | 1937.04 | 1930.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1969.00 | 1969.15 | 1959.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 1969.00 | 1969.15 | 1959.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1981.90 | 1970.84 | 1962.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 1987.00 | 1964.48 | 1962.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 1917.60 | 1955.10 | 1958.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1917.60 | 1955.10 | 1958.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1897.60 | 1930.84 | 1945.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1822.00 | 1816.20 | 1856.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 1822.00 | 1816.20 | 1856.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1839.00 | 1829.52 | 1848.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 1847.90 | 1829.52 | 1848.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1829.20 | 1830.53 | 1845.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 1813.60 | 1826.01 | 1839.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1858.90 | 1829.61 | 1835.94 | SL hit (close>static) qty=1.00 sl=1847.20 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1856.30 | 1842.42 | 1840.88 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1834.60 | 1843.92 | 1844.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1830.00 | 1840.51 | 1843.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 1842.30 | 1840.86 | 1842.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 1842.30 | 1840.86 | 1842.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1842.30 | 1840.86 | 1842.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1842.30 | 1840.86 | 1842.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1830.20 | 1838.73 | 1841.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 1849.10 | 1838.73 | 1841.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1862.90 | 1843.57 | 1843.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 1858.00 | 1843.57 | 1843.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 1856.60 | 1846.17 | 1844.91 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1831.20 | 1846.78 | 1847.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1816.60 | 1837.89 | 1843.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1808.70 | 1802.82 | 1815.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1808.70 | 1802.82 | 1815.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1827.20 | 1808.32 | 1814.87 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1827.10 | 1819.10 | 1818.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 1855.00 | 1829.31 | 1823.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 1836.30 | 1842.86 | 1833.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 1836.30 | 1842.86 | 1833.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1825.00 | 1839.29 | 1832.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1825.00 | 1839.29 | 1832.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1826.60 | 1836.75 | 1832.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 1827.10 | 1836.75 | 1832.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1818.00 | 1829.33 | 1829.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1810.30 | 1825.52 | 1827.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1757.70 | 1750.62 | 1773.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1729.50 | 1743.04 | 1765.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 13:15:00 | 1730.10 | 1742.17 | 1763.32 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:15:00 | 1730.00 | 1737.78 | 1752.73 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1748.60 | 1734.86 | 1745.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 1748.60 | 1734.86 | 1745.04 | SL hit (close>ema400) qty=1.00 sl=1745.04 alert=retest1 |

### Cycle 206 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1759.00 | 1743.35 | 1742.00 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 1734.00 | 1742.59 | 1742.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 1731.80 | 1738.45 | 1740.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1749.60 | 1737.73 | 1739.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1749.60 | 1737.73 | 1739.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1749.60 | 1737.73 | 1739.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1749.60 | 1737.73 | 1739.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1766.50 | 1743.48 | 1742.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1788.00 | 1760.64 | 1751.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1733.10 | 1759.25 | 1752.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 1733.10 | 1759.25 | 1752.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1733.10 | 1759.25 | 1752.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1799.40 | 1766.44 | 1756.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 1793.60 | 1782.30 | 1775.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 1836.00 | 1860.12 | 1862.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1836.00 | 1860.12 | 1862.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 1829.60 | 1854.02 | 1859.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1746.50 | 1738.88 | 1756.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:30:00 | 1747.00 | 1738.88 | 1756.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1748.90 | 1740.33 | 1750.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 1754.80 | 1740.33 | 1750.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1746.40 | 1741.54 | 1750.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 1747.10 | 1741.54 | 1750.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1737.70 | 1732.58 | 1738.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 1734.90 | 1732.58 | 1738.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1735.10 | 1733.09 | 1737.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:30:00 | 1726.50 | 1731.89 | 1736.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1731.50 | 1727.58 | 1732.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 1727.60 | 1728.14 | 1731.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1726.60 | 1725.90 | 1729.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1699.00 | 1720.52 | 1726.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1691.00 | 1720.52 | 1726.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1696.20 | 1694.68 | 1701.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1693.00 | 1694.68 | 1701.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1717.00 | 1703.47 | 1703.75 | SL hit (close>static) qty=1.00 sl=1716.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1729.90 | 1708.76 | 1706.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 1739.00 | 1714.81 | 1709.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1705.60 | 1712.97 | 1708.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1705.60 | 1712.97 | 1708.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1705.60 | 1712.97 | 1708.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1699.30 | 1712.97 | 1708.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1707.50 | 1711.87 | 1708.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1713.80 | 1708.93 | 1708.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 1717.30 | 1722.16 | 1720.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 1713.30 | 1721.31 | 1720.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1712.30 | 1719.51 | 1719.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 1712.30 | 1719.51 | 1719.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1696.00 | 1713.45 | 1716.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 1680.10 | 1675.61 | 1690.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1651.80 | 1675.61 | 1690.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1662.60 | 1648.01 | 1657.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 1662.60 | 1648.01 | 1657.88 | SL hit (close>ema400) qty=1.00 sl=1657.88 alert=retest1 |

### Cycle 212 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 1672.00 | 1658.87 | 1657.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1673.00 | 1662.04 | 1659.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1688.20 | 1689.34 | 1680.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:45:00 | 1691.90 | 1689.34 | 1680.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1684.80 | 1687.74 | 1680.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 1692.60 | 1688.71 | 1681.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 1693.90 | 1689.33 | 1682.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 1697.00 | 1689.18 | 1683.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1670.40 | 1681.23 | 1681.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1670.40 | 1681.23 | 1681.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1654.00 | 1666.69 | 1673.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1664.20 | 1642.57 | 1648.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1664.20 | 1642.57 | 1648.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1664.20 | 1642.57 | 1648.13 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1660.30 | 1651.58 | 1651.42 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 1641.00 | 1649.90 | 1650.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 1629.20 | 1645.76 | 1648.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1572.70 | 1571.52 | 1589.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 1572.70 | 1571.52 | 1589.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1595.80 | 1576.38 | 1590.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1595.70 | 1576.38 | 1590.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1593.30 | 1579.76 | 1590.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 1581.50 | 1580.65 | 1589.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1620.10 | 1590.68 | 1592.87 | SL hit (close>static) qty=1.00 sl=1599.40 alert=retest2 |

### Cycle 216 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1622.20 | 1596.98 | 1595.54 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1572.30 | 1594.76 | 1596.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1563.70 | 1588.55 | 1593.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1567.50 | 1534.53 | 1551.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1567.50 | 1534.53 | 1551.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1567.50 | 1534.53 | 1551.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1546.50 | 1536.92 | 1551.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1574.00 | 1559.20 | 1558.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1574.00 | 1559.20 | 1558.48 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1517.30 | 1552.90 | 1555.87 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1572.00 | 1557.87 | 1556.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1572.90 | 1560.47 | 1557.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1657.00 | 1659.46 | 1641.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 09:15:00 | 1672.40 | 1659.46 | 1641.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1688.10 | 1694.86 | 1674.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1704.70 | 1696.02 | 1676.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 1711.70 | 1697.07 | 1685.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 1745.50 | 1752.56 | 1753.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1745.50 | 1752.56 | 1753.46 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1760.40 | 1753.79 | 1753.73 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 1751.30 | 1753.29 | 1753.51 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1774.00 | 1757.40 | 1755.34 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1734.00 | 1756.56 | 1757.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1726.30 | 1750.51 | 1754.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1741.00 | 1733.02 | 1742.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1741.00 | 1733.02 | 1742.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1741.00 | 1733.02 | 1742.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1743.20 | 1733.02 | 1742.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1742.50 | 1734.91 | 1742.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1747.90 | 1734.91 | 1742.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1754.30 | 1738.79 | 1743.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1754.30 | 1738.79 | 1743.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1756.50 | 1742.33 | 1745.06 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1763.50 | 1748.59 | 1747.56 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 1733.90 | 1746.89 | 1747.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 1731.00 | 1743.71 | 1745.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 1737.40 | 1736.95 | 1741.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:30:00 | 1736.10 | 1736.95 | 1741.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1734.00 | 1736.36 | 1740.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 1732.70 | 1735.63 | 1739.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1748.60 | 1716.51 | 1721.76 | SL hit (close>static) qty=1.00 sl=1741.20 alert=retest2 |

### Cycle 228 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1762.30 | 1725.67 | 1725.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 1766.50 | 1741.37 | 1734.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 1738.40 | 1740.78 | 1734.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 14:00:00 | 1738.40 | 1740.78 | 1734.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1741.20 | 1740.86 | 1735.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 1738.00 | 1740.86 | 1735.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1728.00 | 1738.29 | 1734.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1770.00 | 1738.29 | 1734.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 11:30:00 | 715.00 | 2023-05-18 15:15:00 | 704.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2023-05-26 14:30:00 | 737.80 | 2023-06-08 09:15:00 | 811.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-29 09:15:00 | 756.40 | 2023-06-08 10:15:00 | 832.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-30 09:15:00 | 748.15 | 2023-06-08 10:15:00 | 822.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-21 11:15:00 | 800.50 | 2023-06-27 09:15:00 | 802.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-06-21 14:30:00 | 800.70 | 2023-06-27 09:15:00 | 802.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-06-21 15:15:00 | 799.60 | 2023-06-27 09:15:00 | 802.95 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-06-22 10:45:00 | 800.30 | 2023-06-27 09:15:00 | 802.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-07-06 12:30:00 | 793.40 | 2023-07-10 09:15:00 | 802.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-07-06 13:30:00 | 792.80 | 2023-07-10 09:15:00 | 802.10 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-07-07 10:30:00 | 790.55 | 2023-07-10 09:15:00 | 802.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-07-12 09:15:00 | 821.45 | 2023-07-13 13:15:00 | 801.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2023-07-19 09:45:00 | 865.95 | 2023-07-21 11:15:00 | 853.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-07-19 11:15:00 | 865.40 | 2023-07-21 11:15:00 | 853.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-07-19 12:15:00 | 866.50 | 2023-07-21 11:15:00 | 853.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-07-20 14:00:00 | 867.95 | 2023-07-21 11:15:00 | 853.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-07-24 14:45:00 | 853.85 | 2023-07-25 12:15:00 | 865.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2023-07-25 10:30:00 | 855.85 | 2023-07-25 12:15:00 | 865.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-07-25 11:00:00 | 856.90 | 2023-07-25 12:15:00 | 865.10 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-07-25 11:30:00 | 855.65 | 2023-07-25 12:15:00 | 865.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-08-17 14:30:00 | 1045.35 | 2023-08-18 09:15:00 | 1033.20 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-08-18 11:30:00 | 1046.10 | 2023-08-21 10:15:00 | 1039.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-08-21 13:00:00 | 1045.70 | 2023-08-21 13:15:00 | 1043.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-08-23 15:15:00 | 1025.00 | 2023-08-29 10:15:00 | 1048.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2023-08-24 14:45:00 | 1027.00 | 2023-08-29 10:15:00 | 1048.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-08-25 10:15:00 | 1026.60 | 2023-08-29 10:15:00 | 1048.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-08-25 12:30:00 | 1026.10 | 2023-08-29 10:15:00 | 1048.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-08-28 12:45:00 | 1019.55 | 2023-08-29 10:15:00 | 1048.10 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2023-09-01 09:15:00 | 1077.35 | 2023-09-07 11:15:00 | 1078.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2023-09-14 10:45:00 | 991.80 | 2023-09-14 14:15:00 | 1030.00 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2023-09-14 11:45:00 | 995.20 | 2023-09-14 14:15:00 | 1030.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2023-09-14 13:30:00 | 996.95 | 2023-09-14 14:15:00 | 1030.00 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest1 | 2023-09-25 13:30:00 | 996.55 | 2023-09-26 09:15:00 | 1009.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest1 | 2023-09-25 15:00:00 | 995.95 | 2023-09-26 09:15:00 | 1009.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-10-03 09:30:00 | 1040.00 | 2023-10-09 14:15:00 | 1056.60 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2023-10-04 13:00:00 | 1050.05 | 2023-10-09 14:15:00 | 1056.60 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2023-10-10 14:15:00 | 1060.15 | 2023-10-11 15:15:00 | 1074.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-10-11 12:30:00 | 1059.10 | 2023-10-11 15:15:00 | 1074.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-10-11 13:15:00 | 1060.10 | 2023-10-11 15:15:00 | 1074.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-10-11 13:45:00 | 1059.25 | 2023-10-11 15:15:00 | 1074.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest1 | 2023-10-18 10:30:00 | 1067.85 | 2023-10-18 14:15:00 | 1093.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest1 | 2023-10-18 12:15:00 | 1068.15 | 2023-10-18 14:15:00 | 1093.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2023-10-19 09:30:00 | 1080.80 | 2023-10-23 09:15:00 | 1026.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 10:00:00 | 1080.60 | 2023-10-23 09:15:00 | 1026.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 09:30:00 | 1080.80 | 2023-10-26 09:15:00 | 972.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-19 10:00:00 | 1080.60 | 2023-10-26 09:15:00 | 972.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-08 09:15:00 | 1099.30 | 2023-11-10 10:15:00 | 1209.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-24 11:15:00 | 1226.05 | 2023-11-28 09:15:00 | 1257.55 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2023-11-24 15:15:00 | 1226.00 | 2023-11-28 09:15:00 | 1257.55 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2023-12-01 09:15:00 | 1279.55 | 2023-12-04 13:15:00 | 1245.60 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2023-12-04 13:00:00 | 1263.55 | 2023-12-04 13:15:00 | 1245.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-12-08 11:00:00 | 1213.10 | 2023-12-13 15:15:00 | 1218.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-12-13 15:15:00 | 1218.00 | 2023-12-13 15:15:00 | 1218.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2023-12-20 09:15:00 | 1307.95 | 2023-12-20 14:15:00 | 1287.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-12-21 12:00:00 | 1300.00 | 2023-12-29 14:15:00 | 1353.05 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2024-01-08 10:15:00 | 1409.45 | 2024-01-08 12:15:00 | 1392.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-01-17 14:00:00 | 1484.30 | 2024-01-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-01-18 10:30:00 | 1478.90 | 2024-01-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-01-23 11:30:00 | 1450.95 | 2024-01-23 14:15:00 | 1500.10 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-01-23 13:30:00 | 1448.25 | 2024-01-23 14:15:00 | 1500.10 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-01-25 09:15:00 | 1484.35 | 2024-01-25 10:15:00 | 1461.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-02-02 14:45:00 | 1410.25 | 2024-02-06 09:15:00 | 1435.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-02-05 15:15:00 | 1399.00 | 2024-02-06 09:15:00 | 1435.10 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-02-12 11:00:00 | 1489.15 | 2024-02-21 12:15:00 | 1553.70 | STOP_HIT | 1.00 | 4.33% |
| BUY | retest2 | 2024-02-13 10:30:00 | 1487.00 | 2024-02-21 12:15:00 | 1553.70 | STOP_HIT | 1.00 | 4.49% |
| SELL | retest2 | 2024-02-29 09:15:00 | 1431.45 | 2024-03-01 09:15:00 | 1510.00 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2024-02-29 10:00:00 | 1423.45 | 2024-03-01 09:15:00 | 1510.00 | STOP_HIT | 1.00 | -6.08% |
| SELL | retest2 | 2024-03-15 10:30:00 | 1279.15 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-03-15 11:00:00 | 1273.00 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-03-15 12:30:00 | 1277.50 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-03-18 10:00:00 | 1276.85 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-03-19 11:30:00 | 1271.65 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-03-19 14:15:00 | 1269.95 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-03-20 09:45:00 | 1270.45 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-03-21 10:30:00 | 1271.05 | 2024-03-22 13:15:00 | 1274.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-04-09 13:45:00 | 1291.60 | 2024-04-10 13:15:00 | 1312.35 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-04-09 14:45:00 | 1290.90 | 2024-04-10 13:15:00 | 1312.35 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-04-09 15:15:00 | 1290.50 | 2024-04-10 13:15:00 | 1312.35 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-04-10 10:45:00 | 1290.40 | 2024-04-10 13:15:00 | 1312.35 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-04-29 12:30:00 | 1353.80 | 2024-05-02 10:15:00 | 1342.25 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-04-30 09:45:00 | 1359.65 | 2024-05-02 10:15:00 | 1342.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-04-30 10:30:00 | 1355.45 | 2024-05-02 10:15:00 | 1342.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-04-30 12:00:00 | 1354.35 | 2024-05-02 10:15:00 | 1342.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-05-07 09:15:00 | 1290.40 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-05-07 13:00:00 | 1294.60 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-05-07 14:15:00 | 1287.65 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-05-07 15:15:00 | 1287.00 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-05-09 13:45:00 | 1283.10 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-05-13 09:30:00 | 1281.30 | 2024-05-13 12:15:00 | 1299.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-30 09:15:00 | 1296.60 | 2024-05-30 13:15:00 | 1303.30 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-06-03 10:15:00 | 1288.80 | 2024-06-05 13:15:00 | 1304.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1282.40 | 2024-06-05 13:15:00 | 1304.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-06-19 09:15:00 | 1386.00 | 2024-06-19 13:15:00 | 1400.10 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-16 09:30:00 | 1572.10 | 2024-07-19 09:15:00 | 1465.00 | STOP_HIT | 1.00 | -6.81% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1530.00 | 2024-08-05 12:15:00 | 1546.60 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2024-08-06 14:30:00 | 1547.50 | 2024-08-07 11:15:00 | 1595.65 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-08-14 12:45:00 | 1559.00 | 2024-08-19 09:15:00 | 1714.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 14:00:00 | 1559.05 | 2024-08-19 09:15:00 | 1714.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 15:00:00 | 1560.00 | 2024-08-19 09:15:00 | 1716.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 09:15:00 | 2061.65 | 2024-09-09 12:15:00 | 2014.35 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-09-09 10:30:00 | 2005.00 | 2024-09-09 12:15:00 | 2014.35 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-09-11 10:30:00 | 1995.00 | 2024-09-17 13:15:00 | 1972.95 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2024-09-11 11:30:00 | 1992.30 | 2024-09-17 13:15:00 | 1972.95 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2024-09-12 13:45:00 | 1992.40 | 2024-09-17 13:15:00 | 1972.95 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2024-09-12 14:15:00 | 1994.30 | 2024-09-17 14:15:00 | 2010.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-17 09:30:00 | 1939.70 | 2024-09-17 14:15:00 | 2010.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-09-17 10:00:00 | 1934.60 | 2024-09-17 14:15:00 | 2010.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-09-17 11:30:00 | 1941.05 | 2024-09-17 14:15:00 | 2010.00 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-09-20 15:15:00 | 1930.50 | 2024-09-25 10:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-09-23 09:30:00 | 1923.00 | 2024-09-25 10:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-24 13:00:00 | 1930.00 | 2024-09-25 10:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-09-24 15:15:00 | 1925.00 | 2024-09-25 10:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-09 10:30:00 | 1922.70 | 2024-10-14 10:15:00 | 1919.95 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-10-09 11:45:00 | 1924.20 | 2024-10-14 10:15:00 | 1919.95 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-10-25 09:15:00 | 1836.75 | 2024-10-29 12:15:00 | 1870.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-11-04 12:00:00 | 1991.70 | 2024-11-11 09:15:00 | 2009.60 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-11-04 12:30:00 | 1994.45 | 2024-11-11 09:15:00 | 2009.60 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-11-05 14:15:00 | 2014.70 | 2024-11-11 09:15:00 | 2009.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-11-14 14:45:00 | 1957.15 | 2024-11-18 09:15:00 | 2005.05 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-11-22 14:45:00 | 1957.30 | 2024-11-25 13:15:00 | 1970.75 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-11-25 09:15:00 | 1957.95 | 2024-11-25 13:15:00 | 1970.75 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-12-09 11:15:00 | 2369.30 | 2024-12-13 11:15:00 | 2391.90 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-12-09 14:45:00 | 2375.55 | 2024-12-13 11:15:00 | 2391.90 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-12-10 11:00:00 | 2369.50 | 2024-12-13 11:15:00 | 2391.90 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-12-10 15:00:00 | 2385.85 | 2024-12-13 11:15:00 | 2391.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-12-20 09:30:00 | 2385.40 | 2024-12-23 15:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-12-20 12:00:00 | 2363.10 | 2024-12-23 15:15:00 | 2432.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-12-23 11:45:00 | 2373.05 | 2024-12-23 15:15:00 | 2432.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-12-23 13:45:00 | 2386.50 | 2024-12-23 15:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-01-02 10:30:00 | 2576.95 | 2025-01-06 11:15:00 | 2519.90 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-01-02 14:30:00 | 2557.00 | 2025-01-06 11:15:00 | 2519.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-01-03 12:45:00 | 2555.95 | 2025-01-06 11:15:00 | 2519.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-01-15 09:15:00 | 2307.00 | 2025-01-17 12:15:00 | 2191.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 09:15:00 | 2307.00 | 2025-01-20 13:15:00 | 2214.50 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2025-02-01 15:00:00 | 2099.65 | 2025-02-03 09:15:00 | 2028.60 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-02-07 11:15:00 | 2197.65 | 2025-02-10 09:15:00 | 2136.80 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-02-07 14:30:00 | 2197.00 | 2025-02-10 09:15:00 | 2136.80 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-02-13 09:15:00 | 2035.75 | 2025-02-14 10:15:00 | 1933.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:45:00 | 2030.25 | 2025-02-14 12:15:00 | 1928.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:45:00 | 2027.05 | 2025-02-14 12:15:00 | 1925.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 2023.75 | 2025-02-14 12:15:00 | 1922.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 2035.75 | 2025-02-17 09:15:00 | 1950.20 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-02-13 10:45:00 | 2030.25 | 2025-02-17 09:15:00 | 1950.20 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2025-02-13 12:45:00 | 2027.05 | 2025-02-17 09:15:00 | 1950.20 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-02-13 13:15:00 | 2023.75 | 2025-02-17 09:15:00 | 1950.20 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-03-17 12:30:00 | 1859.00 | 2025-03-18 09:15:00 | 1891.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1780.40 | 2025-04-11 15:15:00 | 1869.00 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-04-11 12:45:00 | 1835.35 | 2025-04-11 15:15:00 | 1869.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-24 09:15:00 | 1882.00 | 2025-04-24 09:15:00 | 1903.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-24 14:30:00 | 1884.40 | 2025-04-28 09:15:00 | 1790.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 14:30:00 | 1884.40 | 2025-04-28 09:15:00 | 1877.60 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2025-04-25 09:15:00 | 1867.40 | 2025-04-28 12:15:00 | 1911.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-04-28 10:30:00 | 1878.70 | 2025-04-28 12:15:00 | 1911.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-05-16 12:45:00 | 1991.40 | 2025-05-20 09:15:00 | 2190.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-29 11:15:00 | 2166.10 | 2025-06-03 09:15:00 | 2184.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-05-30 09:15:00 | 2160.10 | 2025-06-03 09:15:00 | 2184.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-05-30 10:30:00 | 2165.00 | 2025-06-03 09:15:00 | 2184.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-05-30 12:00:00 | 2164.50 | 2025-06-03 10:15:00 | 2179.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-02 09:15:00 | 2114.40 | 2025-06-03 10:15:00 | 2179.70 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-06-02 12:00:00 | 2147.60 | 2025-06-03 10:15:00 | 2179.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-02 14:00:00 | 2146.30 | 2025-06-03 10:15:00 | 2179.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-06-13 10:30:00 | 2112.50 | 2025-06-16 12:15:00 | 2093.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-13 11:30:00 | 2120.00 | 2025-06-16 12:15:00 | 2093.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-06-19 09:30:00 | 2079.30 | 2025-06-24 12:15:00 | 2063.80 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-06-24 09:45:00 | 2063.60 | 2025-06-24 12:15:00 | 2063.80 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-30 09:45:00 | 2164.00 | 2025-07-01 10:15:00 | 2081.60 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-07-11 13:15:00 | 2033.90 | 2025-07-11 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-28 11:15:00 | 2070.00 | 2025-07-29 14:15:00 | 2082.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-28 11:45:00 | 2070.70 | 2025-07-29 14:15:00 | 2082.80 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-28 12:45:00 | 2059.00 | 2025-07-29 14:15:00 | 2082.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-13 11:00:00 | 2130.30 | 2025-08-20 10:15:00 | 2120.80 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-18 10:15:00 | 2128.30 | 2025-08-20 10:15:00 | 2120.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-08-18 12:30:00 | 2123.10 | 2025-08-20 10:15:00 | 2120.80 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-08-20 10:00:00 | 2122.50 | 2025-08-20 10:15:00 | 2120.80 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-08-25 09:15:00 | 2205.00 | 2025-08-25 14:15:00 | 2165.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-25 13:45:00 | 2190.20 | 2025-08-25 14:15:00 | 2165.20 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-29 09:15:00 | 2128.50 | 2025-09-02 15:15:00 | 2125.80 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-04 12:45:00 | 2137.50 | 2025-09-04 14:15:00 | 2116.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-12 10:45:00 | 2268.30 | 2025-09-23 10:15:00 | 2286.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-09-15 09:15:00 | 2278.10 | 2025-09-23 10:15:00 | 2286.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-09-17 09:15:00 | 2268.40 | 2025-09-23 10:15:00 | 2286.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-10-03 09:15:00 | 2045.60 | 2025-10-09 11:15:00 | 2035.70 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-10-03 10:15:00 | 2044.60 | 2025-10-09 11:15:00 | 2035.70 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2025-10-24 12:00:00 | 2025.20 | 2025-10-24 12:15:00 | 2015.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-10-27 13:15:00 | 2001.50 | 2025-10-29 10:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-10-28 14:00:00 | 1997.40 | 2025-10-29 10:15:00 | 2017.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-30 11:00:00 | 2032.00 | 2025-11-03 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-30 15:00:00 | 2031.80 | 2025-11-03 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-31 13:00:00 | 2036.90 | 2025-11-03 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-11-03 09:15:00 | 2044.20 | 2025-11-03 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-11-06 11:00:00 | 2005.00 | 2025-11-06 13:15:00 | 2020.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-06 12:00:00 | 2010.00 | 2025-11-06 13:15:00 | 2020.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-12 10:45:00 | 1941.10 | 2025-11-13 10:15:00 | 1962.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-12 13:45:00 | 1936.20 | 2025-11-13 10:15:00 | 1962.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest1 | 2025-11-17 12:30:00 | 1925.10 | 2025-11-17 13:15:00 | 1936.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-11-17 13:00:00 | 1926.00 | 2025-11-17 13:15:00 | 1936.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1917.80 | 2025-11-21 13:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-11-20 10:45:00 | 1920.00 | 2025-11-21 13:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-20 13:30:00 | 1917.00 | 2025-11-21 13:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1919.00 | 2025-11-21 13:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-24 09:30:00 | 1902.10 | 2025-11-24 14:15:00 | 1928.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1904.70 | 2025-11-26 15:15:00 | 1920.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1899.30 | 2025-11-26 15:15:00 | 1920.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-27 14:15:00 | 1913.10 | 2025-12-01 09:15:00 | 1910.90 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1916.00 | 2025-12-01 09:15:00 | 1910.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-11-28 09:45:00 | 1912.00 | 2025-12-01 09:15:00 | 1910.90 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1925.00 | 2025-12-01 09:15:00 | 1910.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-05 14:45:00 | 1945.50 | 2025-12-08 12:15:00 | 1919.80 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-08 11:30:00 | 1942.20 | 2025-12-08 12:15:00 | 1919.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-26 13:45:00 | 1987.00 | 2025-12-26 14:15:00 | 1917.60 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-01-01 13:30:00 | 1813.60 | 2026-01-02 10:15:00 | 1858.90 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest1 | 2026-01-22 11:30:00 | 1729.50 | 2026-01-23 15:15:00 | 1748.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest1 | 2026-01-22 13:15:00 | 1730.10 | 2026-01-23 15:15:00 | 1748.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2026-01-23 11:15:00 | 1730.00 | 2026-01-23 15:15:00 | 1748.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1700.40 | 2026-01-28 13:15:00 | 1759.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2026-02-01 11:00:00 | 1799.40 | 2026-02-06 12:15:00 | 1836.00 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2026-02-02 14:45:00 | 1793.60 | 2026-02-06 12:15:00 | 1836.00 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2026-02-16 13:30:00 | 1726.50 | 2026-02-23 13:15:00 | 1717.00 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2026-02-17 11:00:00 | 1731.50 | 2026-02-23 14:15:00 | 1729.90 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2026-02-17 12:45:00 | 1727.60 | 2026-02-23 14:15:00 | 1729.90 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1726.60 | 2026-02-23 14:15:00 | 1729.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-02-23 10:15:00 | 1693.00 | 2026-02-23 14:15:00 | 1729.90 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-24 15:15:00 | 1713.80 | 2026-02-27 12:15:00 | 1712.30 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2026-02-27 11:00:00 | 1717.30 | 2026-02-27 12:15:00 | 1712.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-02-27 11:45:00 | 1713.30 | 2026-02-27 12:15:00 | 1712.30 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest1 | 2026-03-04 09:15:00 | 1651.80 | 2026-03-05 15:15:00 | 1662.60 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-03-06 09:30:00 | 1651.80 | 2026-03-06 13:15:00 | 1680.70 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-03-06 10:15:00 | 1650.80 | 2026-03-06 13:15:00 | 1680.70 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1630.10 | 2026-03-09 14:15:00 | 1672.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-12 11:00:00 | 1692.60 | 2026-03-13 10:15:00 | 1670.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-12 12:15:00 | 1693.90 | 2026-03-13 10:15:00 | 1670.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-03-12 13:15:00 | 1697.00 | 2026-03-13 10:15:00 | 1670.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-03-24 14:30:00 | 1581.50 | 2026-03-25 09:15:00 | 1620.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-04-01 11:00:00 | 1546.50 | 2026-04-01 14:15:00 | 1574.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1704.70 | 2026-04-22 09:15:00 | 1745.50 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2026-04-15 09:30:00 | 1711.70 | 2026-04-22 09:15:00 | 1745.50 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2026-04-29 13:00:00 | 1732.70 | 2026-05-04 09:15:00 | 1748.60 | STOP_HIT | 1.00 | -0.92% |
