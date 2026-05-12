# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
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

### Cycle 1 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 983.55 | 922.63 | 922.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 995.00 | 925.17 | 923.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 943.25 | 955.16 | 942.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 947.15 | 955.83 | 944.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 949.20 | 955.83 | 944.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 936.10 | 955.63 | 944.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 936.10 | 955.63 | 944.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 940.50 | 955.48 | 944.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 942.85 | 950.75 | 942.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 944.25 | 950.68 | 942.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 945.25 | 950.01 | 942.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:15:00 | 942.50 | 949.92 | 942.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 937.85 | 949.80 | 942.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 936.90 | 949.80 | 942.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 936.65 | 949.67 | 942.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 936.65 | 949.67 | 942.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 938.90 | 949.24 | 942.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 945.30 | 949.18 | 942.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.34 | 950.64 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 903.10 | 945.77 | 945.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 893.45 | 943.46 | 944.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 863.80 | 862.32 | 889.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 863.80 | 862.32 | 889.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 778.40 | 757.67 | 780.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 778.40 | 757.67 | 780.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 779.45 | 757.88 | 780.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 781.00 | 757.88 | 780.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 779.00 | 758.09 | 780.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 780.00 | 758.09 | 780.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 775.50 | 758.27 | 780.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 771.00 | 758.40 | 780.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 770.75 | 758.40 | 780.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 770.15 | 758.53 | 780.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.19 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 814.50 | 767.78 | 767.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 816.85 | 769.11 | 768.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 852.30 | 853.01 | 820.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:15:00 | 840.60 | 853.01 | 820.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 824.80 | 854.77 | 826.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 824.80 | 854.77 | 826.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 814.20 | 854.36 | 826.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 814.20 | 854.36 | 826.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 816.00 | 852.48 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 817.60 | 852.48 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 826.95 | 852.23 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:30:00 | 817.40 | 852.23 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 835.30 | 852.06 | 826.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 14:30:00 | 837.90 | 850.20 | 826.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 810.30 | 849.62 | 826.74 | SL hit (close<static) qty=1.00 sl=825.55 alert=retest2 |


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
