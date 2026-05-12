# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1272.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 0 |
| ALERT3 | 78 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 86 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 74
- **Target hits / Stop hits / Partials:** 8 / 78 / 0
- **Avg / median % per leg:** -0.62% / -1.55%
- **Sum % (uncompounded):** -53.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 12 | 15.4% | 8 | 70 | 0 | -0.53% | -41.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 78 | 12 | 15.4% | 8 | 70 | 0 | -0.53% | -41.1% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.56% | -12.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.56% | -12.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 86 | 12 | 14.0% | 8 | 78 | 0 | -0.62% | -53.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 755.55 | 785.45 | 785.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 748.15 | 780.86 | 782.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 759.45 | 758.65 | 768.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-13 10:00:00 | 759.45 | 758.65 | 768.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 777.60 | 758.85 | 767.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:45:00 | 779.55 | 758.85 | 767.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 771.30 | 759.88 | 767.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 11:00:00 | 771.30 | 759.88 | 767.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 770.85 | 759.99 | 767.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 12:15:00 | 771.55 | 759.99 | 767.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 767.35 | 760.24 | 768.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:45:00 | 772.10 | 760.24 | 768.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 769.30 | 760.33 | 768.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 769.85 | 760.33 | 768.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 770.35 | 760.43 | 768.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 12:15:00 | 768.80 | 760.63 | 768.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 15:15:00 | 766.50 | 760.90 | 768.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 09:30:00 | 768.90 | 761.02 | 768.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 10:30:00 | 768.40 | 761.08 | 768.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 772.05 | 761.34 | 767.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 774.20 | 761.34 | 767.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 779.20 | 761.52 | 768.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-21 10:15:00 | 779.20 | 761.52 | 768.03 | SL hit (close>static) qty=1.00 sl=774.90 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 815.40 | 772.53 | 772.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 824.20 | 776.21 | 774.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 837.30 | 838.07 | 815.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-03 15:00:00 | 837.30 | 838.07 | 815.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 815.70 | 835.93 | 817.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 816.65 | 835.93 | 817.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 822.35 | 835.80 | 817.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 12:00:00 | 824.40 | 835.69 | 817.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 13:00:00 | 824.00 | 835.57 | 817.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 14:30:00 | 823.35 | 834.75 | 818.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 829.60 | 834.63 | 818.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 816.60 | 832.81 | 819.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 12:00:00 | 816.60 | 832.81 | 819.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 815.50 | 832.63 | 819.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 813.60 | 832.29 | 819.08 | SL hit (close<static) qty=1.00 sl=815.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 14:15:00 | 785.90 | 816.56 | 816.61 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 870.15 | 816.05 | 815.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 880.15 | 821.30 | 818.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 840.20 | 843.51 | 832.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-19 11:00:00 | 840.20 | 843.51 | 832.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 853.30 | 863.57 | 847.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 852.80 | 863.57 | 847.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 850.50 | 863.05 | 848.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:30:00 | 849.00 | 863.05 | 848.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 843.10 | 862.74 | 848.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:45:00 | 842.45 | 862.74 | 848.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 841.50 | 862.53 | 848.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 841.50 | 862.53 | 848.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 847.10 | 861.71 | 848.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:45:00 | 847.45 | 861.71 | 848.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 849.60 | 861.59 | 848.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 14:45:00 | 853.15 | 861.26 | 848.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 839.20 | 860.96 | 848.48 | SL hit (close<static) qty=1.00 sl=847.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 923.80 | 968.01 | 968.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 919.75 | 966.67 | 967.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 924.80 | 923.64 | 939.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-21 09:30:00 | 931.85 | 923.64 | 939.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 932.45 | 923.99 | 937.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 938.75 | 923.99 | 937.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 946.40 | 924.21 | 937.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 946.40 | 924.21 | 937.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 946.65 | 924.43 | 937.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 948.50 | 924.43 | 937.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 939.05 | 923.75 | 936.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 939.05 | 923.75 | 936.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 933.70 | 923.85 | 936.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:30:00 | 924.35 | 928.03 | 937.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 930.50 | 928.08 | 937.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 925.00 | 928.18 | 937.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 941.85 | 928.54 | 936.80 | SL hit (close>static) qty=1.00 sl=940.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 15:15:00 | 962.85 | 942.69 | 942.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 966.85 | 943.67 | 943.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.45 | 950.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:00:00 | 955.00 | 955.45 | 950.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 945.55 | 955.32 | 950.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 945.55 | 955.32 | 950.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 944.95 | 955.22 | 950.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:00:00 | 944.95 | 955.22 | 950.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 954.00 | 955.16 | 950.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 961.00 | 955.16 | 950.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 09:15:00 | 1057.10 | 989.95 | 972.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.81 | 1135.00 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1187.00 | 1131.48 | 1131.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 1188.50 | 1132.05 | 1131.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.60 | 1231.96 | 1204.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1225.60 | 1231.96 | 1204.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.53 | 1204.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 1202.80 | 1231.53 | 1204.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1212.80 | 1231.34 | 1204.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 1215.70 | 1231.34 | 1204.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 1217.90 | 1231.21 | 1204.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 1220.10 | 1230.79 | 1204.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1120.50 | 1191.41 | 1191.63 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1253.30 | 1187.35 | 1186.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-17 12:15:00 | 768.80 | 2023-11-21 10:15:00 | 779.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-11-17 15:15:00 | 766.50 | 2023-11-21 10:15:00 | 779.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-11-20 09:30:00 | 768.90 | 2023-11-21 10:15:00 | 779.20 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-11-20 10:30:00 | 768.40 | 2023-11-21 10:15:00 | 779.20 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-11-23 10:15:00 | 769.50 | 2023-11-24 09:15:00 | 783.20 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-01-10 12:00:00 | 824.40 | 2024-01-17 14:15:00 | 813.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-01-10 13:00:00 | 824.00 | 2024-01-17 14:15:00 | 813.60 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-01-11 14:30:00 | 823.35 | 2024-01-17 14:15:00 | 813.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-01-12 09:15:00 | 829.60 | 2024-01-17 14:15:00 | 813.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-01-19 14:30:00 | 819.25 | 2024-01-20 09:15:00 | 813.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-01-30 09:15:00 | 819.90 | 2024-01-30 14:15:00 | 813.15 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-01-31 09:30:00 | 820.90 | 2024-01-31 12:15:00 | 813.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-01-31 15:00:00 | 819.25 | 2024-02-01 09:15:00 | 812.55 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-02-02 14:15:00 | 823.75 | 2024-02-06 09:15:00 | 810.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-02-05 10:30:00 | 822.40 | 2024-02-06 09:15:00 | 810.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-02-05 11:30:00 | 825.60 | 2024-02-06 09:15:00 | 810.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-02-05 12:45:00 | 822.15 | 2024-02-06 09:15:00 | 810.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-02-07 09:15:00 | 829.15 | 2024-02-09 10:15:00 | 805.50 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-02-08 15:00:00 | 822.25 | 2024-02-09 10:15:00 | 805.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-02-16 10:15:00 | 821.95 | 2024-02-20 10:15:00 | 813.65 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-02-16 11:30:00 | 821.95 | 2024-02-20 10:15:00 | 813.65 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-02-20 12:45:00 | 819.25 | 2024-02-26 10:15:00 | 810.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-02-20 15:00:00 | 821.45 | 2024-02-26 10:15:00 | 810.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-02-27 15:00:00 | 819.45 | 2024-02-28 10:15:00 | 810.20 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-03-01 09:30:00 | 820.45 | 2024-03-06 09:15:00 | 806.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-03-07 09:15:00 | 841.55 | 2024-03-13 09:15:00 | 813.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-03-07 12:30:00 | 829.35 | 2024-03-13 09:15:00 | 813.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-03-11 11:15:00 | 827.00 | 2024-03-13 09:15:00 | 813.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-03-11 12:30:00 | 828.50 | 2024-03-13 09:15:00 | 813.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-10 14:45:00 | 853.15 | 2024-05-13 09:15:00 | 839.20 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-05-13 12:45:00 | 852.35 | 2024-06-04 11:15:00 | 834.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-05-13 13:30:00 | 853.40 | 2024-06-04 11:15:00 | 834.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-06-04 12:45:00 | 856.10 | 2024-06-04 14:15:00 | 847.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-06 09:15:00 | 887.00 | 2024-07-29 12:15:00 | 898.95 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-06-06 13:15:00 | 884.80 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2024-07-22 09:30:00 | 882.60 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2024-07-23 12:30:00 | 883.25 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2024-07-29 11:15:00 | 904.50 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-30 12:15:00 | 903.90 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2024-07-30 14:30:00 | 904.35 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2024-07-31 09:15:00 | 907.20 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2024-07-31 10:30:00 | 919.10 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest2 | 2024-08-12 09:30:00 | 915.80 | 2024-08-14 10:15:00 | 893.40 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-08-12 10:00:00 | 917.85 | 2024-08-14 10:15:00 | 893.40 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-08-19 13:00:00 | 915.00 | 2024-09-24 09:15:00 | 1006.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-04 11:30:00 | 931.60 | 2024-09-30 09:15:00 | 1021.52 | TARGET_HIT | 1.00 | 9.65% |
| BUY | retest2 | 2024-09-04 13:15:00 | 928.65 | 2024-09-30 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2024-09-06 11:45:00 | 930.90 | 2024-09-30 09:15:00 | 1023.17 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2024-09-09 12:30:00 | 930.15 | 2024-09-30 10:15:00 | 1024.76 | TARGET_HIT | 1.00 | 10.17% |
| BUY | retest2 | 2024-10-28 15:15:00 | 972.15 | 2024-10-29 09:15:00 | 958.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-11-01 18:15:00 | 971.40 | 2024-11-04 09:15:00 | 954.70 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-05 09:15:00 | 971.85 | 2024-11-12 12:15:00 | 964.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-05 09:45:00 | 973.00 | 2024-11-12 12:15:00 | 964.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-11-26 14:30:00 | 964.65 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-11-27 09:45:00 | 965.65 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-27 12:45:00 | 963.15 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-11-29 09:45:00 | 962.20 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-02 10:30:00 | 977.00 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-12-02 11:15:00 | 977.55 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2024-12-13 11:15:00 | 977.10 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-02-01 12:30:00 | 924.35 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-02-01 14:15:00 | 930.50 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-02-03 09:15:00 | 925.00 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-03-03 09:15:00 | 961.00 | 2025-03-21 09:15:00 | 1057.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 960.95 | 2025-04-08 09:15:00 | 943.95 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-04-08 12:45:00 | 955.35 | 2025-04-09 09:15:00 | 942.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-04-08 15:15:00 | 956.20 | 2025-04-09 09:15:00 | 942.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-04-11 11:15:00 | 990.25 | 2025-05-02 11:15:00 | 968.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-04-11 14:30:00 | 989.40 | 2025-05-02 11:15:00 | 968.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-12 12:30:00 | 994.50 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-05-13 15:15:00 | 989.70 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1010.50 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-22 09:30:00 | 993.20 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-22 10:30:00 | 996.10 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-30 15:15:00 | 994.00 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-09 11:15:00 | 1012.75 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-06-09 11:45:00 | 1011.90 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1013.00 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1013.50 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1007.70 | 2025-06-23 09:15:00 | 987.90 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1010.60 | 2025-09-08 09:15:00 | 1111.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1014.55 | 2025-09-16 12:15:00 | 1116.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 13:15:00 | 1215.70 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-03-04 14:00:00 | 1217.90 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-05 09:15:00 | 1220.10 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-10 12:45:00 | 1213.00 | 2026-03-11 09:15:00 | 1195.30 | STOP_HIT | 1.00 | -1.46% |
