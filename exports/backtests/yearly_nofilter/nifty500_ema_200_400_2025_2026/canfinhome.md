# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 878.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 22 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 21
- **Target hits / Stop hits / Partials:** 1 / 23 / 3
- **Avg / median % per leg:** -0.74% / -1.50%
- **Sum % (uncompounded):** -19.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 0 | 16 | 2 | -1.04% | -18.7% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.98% | 11.9% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.19% | -30.6% |
| SELL (all) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.14% | -1.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.14% | -1.3% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.98% | 11.9% |
| retest2 (combined) | 23 | 2 | 8.7% | 1 | 21 | 1 | -1.39% | -31.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 11:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:45:00 | 783.15 | 769.72 | 745.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 11:30:00 | 783.55 | 772.14 | 748.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:15:00 | 822.31 | 788.52 | 763.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:15:00 | 822.73 | 788.52 | 763.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 790.85 | 791.12 | 766.96 | SL hit (close<ema200) qty=0.50 sl=791.12 alert=retest1 |

### Cycle 2 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 759.40 | 765.48 | 765.51 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 778.10 | 765.57 | 765.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 778.45 | 765.79 | 765.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 766.30 | 767.98 | 766.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 766.30 | 767.98 | 766.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 766.30 | 767.98 | 766.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 766.30 | 767.98 | 766.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 765.00 | 767.95 | 766.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 758.20 | 767.95 | 766.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 751.00 | 767.66 | 766.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 751.55 | 767.66 | 766.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 729.75 | 765.53 | 765.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 727.00 | 765.15 | 765.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 15:15:00 | 759.00 | 757.37 | 761.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 758.70 | 757.37 | 761.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 768.05 | 757.48 | 761.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 768.05 | 757.48 | 761.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 764.50 | 757.55 | 761.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 764.00 | 757.72 | 761.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 770.95 | 758.09 | 761.48 | SL hit (close>static) qty=1.00 sl=768.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 779.50 | 761.74 | 761.72 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 740.45 | 761.59 | 761.65 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 793.20 | 761.85 | 761.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 796.90 | 764.12 | 762.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 900.10 | 908.31 | 879.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 900.10 | 908.31 | 879.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 894.00 | 918.42 | 892.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 891.30 | 918.29 | 892.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 889.25 | 917.52 | 892.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 889.25 | 917.52 | 892.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 894.45 | 917.29 | 892.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 882.90 | 917.29 | 892.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 896.05 | 917.08 | 892.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 881.75 | 917.08 | 892.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 880.50 | 916.71 | 892.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 880.90 | 916.71 | 892.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 881.50 | 916.36 | 892.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:45:00 | 875.70 | 916.36 | 892.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 894.85 | 914.70 | 892.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:30:00 | 892.85 | 914.70 | 892.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 889.30 | 914.45 | 892.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 895.80 | 913.42 | 892.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 898.85 | 912.77 | 892.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 896.40 | 913.29 | 893.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:00:00 | 896.95 | 914.01 | 895.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 901.60 | 913.89 | 895.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 914.60 | 913.38 | 895.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 907.10 | 913.24 | 895.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 13:30:00 | 902.85 | 912.79 | 896.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 886.20 | 912.22 | 896.27 | SL hit (close<static) qty=1.00 sl=894.70 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 828.80 | 898.33 | 898.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 817.50 | 894.58 | 896.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 876.65 | 874.52 | 884.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 876.65 | 874.52 | 884.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 873.30 | 866.63 | 879.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 873.30 | 866.63 | 879.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 884.00 | 866.81 | 879.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 884.00 | 866.81 | 879.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 873.00 | 866.87 | 879.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:15:00 | 868.05 | 866.87 | 879.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:15:00 | 824.65 | 864.39 | 877.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 781.25 | 860.90 | 875.16 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 881.45 | 867.02 | 866.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 901.55 | 868.84 | 867.90 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 11:45:00 | 783.15 | 2025-07-09 10:15:00 | 822.31 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-26 11:30:00 | 783.55 | 2025-07-09 10:15:00 | 822.73 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-24 11:45:00 | 783.15 | 2025-07-11 11:15:00 | 790.85 | STOP_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-06-26 11:30:00 | 783.55 | 2025-07-11 11:15:00 | 790.85 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2025-09-03 13:15:00 | 764.00 | 2025-09-04 09:15:00 | 770.95 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-04 10:15:00 | 764.10 | 2025-09-16 09:15:00 | 774.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-04 12:15:00 | 762.45 | 2025-09-16 09:15:00 | 774.90 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-23 09:45:00 | 764.30 | 2025-09-23 12:15:00 | 774.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-13 15:15:00 | 895.80 | 2026-01-27 09:15:00 | 886.20 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-14 12:00:00 | 898.85 | 2026-01-27 09:15:00 | 886.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-19 10:00:00 | 896.40 | 2026-01-27 09:15:00 | 886.20 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-01-21 12:00:00 | 896.95 | 2026-01-27 10:15:00 | 883.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-22 09:15:00 | 914.60 | 2026-01-27 10:15:00 | 883.50 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-01-22 11:15:00 | 907.10 | 2026-01-27 10:15:00 | 883.50 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-01-23 13:30:00 | 902.85 | 2026-01-27 10:15:00 | 883.50 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2026-01-28 11:15:00 | 903.35 | 2026-02-01 15:15:00 | 893.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-01-30 14:15:00 | 909.55 | 2026-02-12 14:15:00 | 886.90 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-02 09:15:00 | 934.45 | 2026-02-12 14:15:00 | 886.90 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2026-02-02 15:00:00 | 912.65 | 2026-02-12 14:15:00 | 886.90 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-02-18 09:30:00 | 911.95 | 2026-02-19 13:15:00 | 897.05 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-02-19 11:15:00 | 902.45 | 2026-02-19 15:15:00 | 886.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-23 15:00:00 | 908.00 | 2026-02-24 09:15:00 | 886.45 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-03-18 13:15:00 | 868.05 | 2026-03-19 14:15:00 | 824.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 13:15:00 | 868.05 | 2026-03-23 09:15:00 | 781.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-20 09:15:00 | 868.45 | 2026-04-21 09:15:00 | 901.80 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2026-04-20 15:15:00 | 864.90 | 2026-04-21 09:15:00 | 901.80 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-04-30 09:15:00 | 864.85 | 2026-05-04 09:15:00 | 889.80 | STOP_HIT | 1.00 | -2.88% |
