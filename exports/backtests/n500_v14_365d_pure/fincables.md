# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 1144.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 3 / 14 / 2
- **Avg / median % per leg:** 1.31% / -1.08%
- **Sum % (uncompounded):** 24.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 3 | 6 | 0 | 2.30% | 20.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 3 | 6 | 0 | 2.30% | 20.7% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 8 | 2 | 0.42% | 4.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 0 | 8 | 2 | 0.42% | 4.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 6 | 31.6% | 3 | 14 | 2 | 1.31% | 24.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 959.55 | 912.17 | 912.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 978.90 | 912.84 | 912.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 947.00 | 955.13 | 939.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 943.25 | 955.13 | 939.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 936.10 | 955.61 | 941.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 936.10 | 955.61 | 941.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 940.50 | 955.46 | 941.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 942.85 | 950.73 | 940.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 944.25 | 950.67 | 940.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 945.25 | 950.00 | 940.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:15:00 | 942.50 | 949.91 | 940.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 937.85 | 949.79 | 940.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 936.90 | 949.79 | 940.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 936.65 | 949.66 | 940.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 936.65 | 949.66 | 940.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 938.90 | 949.22 | 940.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 945.30 | 949.17 | 940.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.33 | 949.34 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.33 | 949.34 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.33 | 949.34 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.33 | 949.34 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.33 | 949.34 | SL hit (close<static) qty=1.00 sl=935.15 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 893.45 | 943.46 | 943.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 893.00 | 942.96 | 943.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 863.80 | 862.32 | 888.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 863.80 | 862.32 | 888.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 778.40 | 757.67 | 780.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 778.40 | 757.67 | 780.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 779.45 | 757.88 | 780.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 781.00 | 757.88 | 780.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 779.00 | 758.09 | 780.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 780.00 | 758.09 | 780.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 775.50 | 758.27 | 780.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 771.00 | 758.40 | 780.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 770.75 | 758.40 | 780.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 770.15 | 758.53 | 780.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.15 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.15 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.15 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 770.15 | 764.10 | 780.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 731.64 | 762.94 | 777.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 770.85 | 760.42 | 775.61 | SL hit (close>ema200) qty=0.50 sl=760.42 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 785.45 | 760.67 | 775.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 785.45 | 760.67 | 775.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 785.40 | 760.92 | 775.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:30:00 | 787.15 | 760.92 | 775.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 782.70 | 763.39 | 776.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 782.70 | 763.39 | 776.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 780.80 | 767.27 | 776.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 780.80 | 767.27 | 776.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 779.20 | 767.56 | 776.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 777.30 | 767.56 | 776.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 777.00 | 767.65 | 776.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 779.20 | 767.65 | 776.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 773.25 | 767.71 | 776.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:45:00 | 770.50 | 767.73 | 776.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 772.80 | 766.66 | 775.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 780.00 | 767.38 | 775.31 | SL hit (close>static) qty=1.00 sl=779.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 780.00 | 767.38 | 775.31 | SL hit (close>static) qty=1.00 sl=779.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 772.05 | 767.67 | 775.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 733.45 | 765.93 | 774.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 743.95 | 742.89 | 757.82 | SL hit (close>ema200) qty=0.50 sl=742.89 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 15:15:00 | 769.80 | 744.10 | 757.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 784.50 | 745.06 | 757.37 | SL hit (close>static) qty=1.00 sl=779.60 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 814.50 | 767.78 | 767.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 816.85 | 769.11 | 768.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 852.30 | 853.01 | 820.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:15:00 | 840.60 | 853.01 | 820.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 824.80 | 854.77 | 826.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 824.80 | 854.77 | 826.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 814.20 | 854.36 | 826.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 814.20 | 854.36 | 826.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 816.00 | 852.48 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 817.60 | 852.48 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 826.95 | 852.23 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:30:00 | 817.40 | 852.23 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 835.30 | 852.06 | 826.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 14:30:00 | 837.90 | 850.20 | 826.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 810.30 | 849.62 | 826.74 | SL hit (close<static) qty=1.00 sl=825.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 838.30 | 832.99 | 821.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 844.20 | 832.99 | 821.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 841.15 | 836.00 | 824.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 922.13 | 840.01 | 827.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-16 09:15:00 | 928.62 | 840.01 | 827.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-16 09:15:00 | 925.27 | 840.01 | 827.36 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-25 09:15:00 | 942.85 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-25 09:45:00 | 944.25 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-26 09:15:00 | 945.25 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-26 10:15:00 | 942.50 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-06-27 09:30:00 | 945.30 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-17 10:30:00 | 771.00 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-17 11:15:00 | 770.75 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-17 12:15:00 | 770.15 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-23 14:30:00 | 770.15 | 2025-12-30 11:15:00 | 731.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:30:00 | 770.15 | 2026-01-01 11:15:00 | 770.85 | STOP_HIT | 0.50 | -0.09% |
| SELL | retest2 | 2026-01-09 11:45:00 | 770.50 | 2026-01-16 13:15:00 | 780.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-14 13:45:00 | 772.80 | 2026-01-16 13:15:00 | 780.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-19 10:00:00 | 772.05 | 2026-01-20 13:15:00 | 733.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:00:00 | 772.05 | 2026-02-04 09:15:00 | 743.95 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2026-02-06 15:15:00 | 769.80 | 2026-02-09 10:15:00 | 784.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-25 14:30:00 | 837.90 | 2026-03-27 09:15:00 | 810.30 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-04-08 09:30:00 | 838.30 | 2026-04-16 09:15:00 | 922.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 10:15:00 | 844.20 | 2026-04-16 09:15:00 | 928.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 841.15 | 2026-04-16 09:15:00 | 925.27 | TARGET_HIT | 1.00 | 10.00% |
