# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 41
- **Target hits / Stop hits / Partials:** 4 / 41 / 0
- **Avg / median % per leg:** -0.69% / -1.44%
- **Sum % (uncompounded):** -31.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 4 | 17.4% | 4 | 19 | 0 | 0.64% | 14.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 4 | 17.4% | 4 | 19 | 0 | 0.64% | 14.7% |
| SELL (all) | 22 | 0 | 0.0% | 0 | 22 | 0 | -2.08% | -45.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 0 | 0.0% | 0 | 22 | 0 | -2.08% | -45.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 4 | 8.9% | 4 | 41 | 0 | -0.69% | -31.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 726.23 | 689.86 | 689.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 727.80 | 691.17 | 690.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 700.69 | 702.71 | 697.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 11:00:00 | 700.69 | 702.71 | 697.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 706.25 | 708.45 | 702.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 13:45:00 | 707.30 | 707.05 | 702.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 10:45:00 | 708.57 | 706.99 | 702.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:15:00 | 706.90 | 706.99 | 702.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 707.18 | 706.93 | 702.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 706.00 | 706.90 | 702.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 701.04 | 706.90 | 702.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 700.00 | 706.83 | 702.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 700.00 | 706.83 | 702.53 | SL hit (close<static) qty=1.00 sl=702.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 663.45 | 699.12 | 699.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 661.00 | 691.08 | 694.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 675.99 | 674.14 | 683.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 14:00:00 | 675.99 | 674.14 | 683.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 682.58 | 674.45 | 682.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 682.58 | 674.45 | 682.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 683.06 | 674.54 | 682.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 682.33 | 674.54 | 682.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 683.90 | 674.63 | 682.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:15:00 | 683.32 | 674.63 | 682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 686.83 | 674.75 | 682.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 686.83 | 674.75 | 682.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 686.50 | 674.87 | 682.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 687.40 | 674.87 | 682.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 731.20 | 688.66 | 688.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 735.27 | 694.11 | 691.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 729.67 | 739.74 | 721.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 729.67 | 739.74 | 721.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 729.67 | 739.74 | 721.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 728.71 | 739.74 | 721.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 719.30 | 739.35 | 721.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 719.30 | 739.35 | 721.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 722.10 | 739.18 | 721.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 717.90 | 739.18 | 721.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 722.10 | 739.01 | 721.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 724.32 | 739.01 | 721.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 11:00:00 | 724.82 | 738.74 | 721.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 13:45:00 | 724.50 | 738.32 | 721.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 723.71 | 737.60 | 721.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 722.61 | 737.31 | 721.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 722.61 | 737.31 | 721.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 718.99 | 737.12 | 721.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 718.99 | 737.12 | 721.98 | SL hit (close<static) qty=1.00 sl=720.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 691.10 | 714.11 | 714.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 684.30 | 713.59 | 713.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 679.20 | 676.70 | 688.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 679.20 | 676.70 | 688.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 686.63 | 676.95 | 688.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 687.50 | 676.95 | 688.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 691.95 | 677.29 | 688.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:00:00 | 691.95 | 677.29 | 688.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 688.40 | 677.40 | 688.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:30:00 | 691.33 | 677.40 | 688.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 685.47 | 677.72 | 688.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:45:00 | 682.59 | 690.47 | 692.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 682.74 | 690.39 | 692.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:15:00 | 681.01 | 690.39 | 692.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:00:00 | 682.61 | 690.32 | 692.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 695.06 | 689.60 | 692.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 695.06 | 689.60 | 692.30 | SL hit (close>static) qty=1.00 sl=692.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 743.62 | 694.22 | 694.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 755.91 | 696.16 | 695.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 849.50 | 859.85 | 827.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 11:00:00 | 849.50 | 859.85 | 827.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 861.35 | 889.77 | 859.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 867.40 | 889.77 | 859.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 860.35 | 889.47 | 859.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 860.90 | 889.47 | 859.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 858.50 | 889.17 | 859.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 859.20 | 889.17 | 859.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 861.55 | 888.89 | 859.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 13:30:00 | 864.10 | 888.65 | 859.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 864.30 | 888.13 | 859.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:45:00 | 867.85 | 887.93 | 859.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:00:00 | 863.55 | 886.58 | 863.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 866.20 | 886.38 | 863.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 894.20 | 886.38 | 863.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-09 09:15:00 | 950.51 | 907.78 | 889.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 872.05 | 911.16 | 911.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 860.50 | 904.26 | 907.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 915.00 | 896.50 | 903.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 912.55 | 896.66 | 903.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 908.80 | 897.11 | 903.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 936.50 | 894.07 | 899.57 | SL hit (close>static) qty=1.00 sl=918.75 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 948.95 | 904.73 | 904.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 961.00 | 905.71 | 905.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1011.90 | 1040.32 | 1007.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1002.70 | 1039.94 | 1007.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1002.70 | 1039.94 | 1007.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1001.10 | 1039.56 | 1007.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1009.40 | 1038.50 | 1007.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 1006.70 | 1035.04 | 1007.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1008.40 | 1034.21 | 1007.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:30:00 | 1007.20 | 1027.77 | 1009.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 994.80 | 1026.90 | 1009.08 | SL hit (close<static) qty=1.00 sl=997.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 975.00 | 1008.84 | 1008.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 968.45 | 1003.44 | 1006.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 955.95 | 954.92 | 974.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:00:00 | 955.95 | 954.92 | 974.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 969.95 | 955.52 | 974.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 972.00 | 956.66 | 973.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 977.90 | 956.66 | 973.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 972.05 | 956.81 | 973.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:00:00 | 964.50 | 959.49 | 973.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 964.70 | 959.54 | 973.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 965.30 | 959.60 | 973.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:00:00 | 965.50 | 959.84 | 973.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 970.50 | 960.03 | 973.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 972.00 | 960.03 | 973.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.13 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 925.45 | 983.72 | 984.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.10 | 920.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 921.70 | 885.46 | 920.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 918.10 | 885.79 | 920.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 908.80 | 886.96 | 920.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 898.85 | 890.44 | 920.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 912.50 | 892.21 | 919.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 912.50 | 892.64 | 919.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-06 11:30:00 | 694.86 | 2024-06-07 09:15:00 | 712.50 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-07-15 13:45:00 | 707.30 | 2024-07-18 10:15:00 | 700.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-07-16 10:45:00 | 708.57 | 2024-07-18 10:15:00 | 700.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-07-16 11:15:00 | 706.90 | 2024-07-18 10:15:00 | 700.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-07-16 14:30:00 | 707.18 | 2024-07-18 10:15:00 | 700.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-10-07 09:15:00 | 724.32 | 2024-10-08 14:15:00 | 718.99 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-07 11:00:00 | 724.82 | 2024-10-08 14:15:00 | 718.99 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-10-07 13:45:00 | 724.50 | 2024-10-08 14:15:00 | 718.99 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-10-08 12:15:00 | 723.71 | 2024-10-08 14:15:00 | 718.99 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-10-09 09:15:00 | 733.96 | 2024-10-15 09:15:00 | 712.11 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-10-14 12:30:00 | 722.50 | 2024-10-15 09:15:00 | 712.11 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-14 13:45:00 | 722.97 | 2024-10-15 09:15:00 | 712.11 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-10-14 14:30:00 | 722.62 | 2024-10-15 09:15:00 | 712.11 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-10-15 09:15:00 | 725.00 | 2024-10-15 09:15:00 | 712.11 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-12-24 10:45:00 | 682.59 | 2024-12-27 09:15:00 | 695.06 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-24 11:30:00 | 682.74 | 2024-12-27 09:15:00 | 695.06 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-12-24 12:15:00 | 681.01 | 2024-12-27 09:15:00 | 695.06 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-12-24 13:00:00 | 682.61 | 2024-12-27 09:15:00 | 695.06 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-12-30 13:30:00 | 689.99 | 2025-01-02 09:15:00 | 716.51 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-01-01 12:45:00 | 690.00 | 2025-01-02 09:15:00 | 716.51 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-04-30 13:30:00 | 864.10 | 2025-06-09 09:15:00 | 950.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:15:00 | 864.30 | 2025-06-09 09:15:00 | 950.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:45:00 | 867.85 | 2025-06-09 09:15:00 | 954.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 15:00:00 | 863.55 | 2025-06-09 09:15:00 | 949.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 894.20 | 2025-08-07 09:15:00 | 872.05 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-18 13:45:00 | 908.80 | 2025-09-04 09:15:00 | 936.50 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-11-11 15:15:00 | 1009.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-13 15:15:00 | 1006.70 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1008.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-24 10:30:00 | 1007.20 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1022.90 | 2025-12-24 15:15:00 | 1009.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-10 10:00:00 | 964.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-10 11:00:00 | 964.70 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-10 12:00:00 | 965.30 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-10 15:00:00 | 965.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-09 09:15:00 | 908.80 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 09:15:00 | 898.85 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-04-15 13:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-15 15:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-16 11:15:00 | 920.50 | 2026-04-21 11:15:00 | 939.75 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-23 09:30:00 | 920.05 | 2026-04-24 15:15:00 | 923.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-04-23 10:00:00 | 919.85 | 2026-04-27 13:15:00 | 926.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-23 14:00:00 | 921.70 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-23 15:15:00 | 916.50 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-04-27 10:00:00 | 913.95 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.46% |
