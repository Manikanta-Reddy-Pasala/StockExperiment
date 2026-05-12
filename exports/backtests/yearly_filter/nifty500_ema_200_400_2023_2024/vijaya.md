# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1275.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 11 |
| TARGET_HIT | 11 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 31
- **Target hits / Stop hits / Partials:** 11 / 32 / 11
- **Avg / median % per leg:** 1.14% / -1.12%
- **Sum % (uncompounded):** 61.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.39% | 8.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.39% | 8.3% |
| SELL (all) | 48 | 21 | 43.8% | 9 | 28 | 11 | 1.11% | 53.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 21 | 43.8% | 9 | 28 | 11 | 1.11% | 53.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 23 | 42.6% | 11 | 32 | 11 | 1.14% | 61.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 12:15:00 | 608.60 | 633.00 | 633.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 13:15:00 | 605.35 | 632.73 | 632.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 15:15:00 | 629.00 | 628.68 | 630.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 15:15:00 | 629.00 | 628.68 | 630.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 629.00 | 628.68 | 630.77 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 13:15:00 | 659.00 | 632.33 | 632.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 664.15 | 635.50 | 633.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 10:15:00 | 665.45 | 668.19 | 655.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 11:00:00 | 665.45 | 668.19 | 655.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 665.40 | 668.27 | 656.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:15:00 | 671.65 | 668.24 | 656.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:45:00 | 673.55 | 668.29 | 656.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-10 09:15:00 | 738.82 | 672.48 | 659.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 907.05 | 1061.18 | 1061.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 882.05 | 1056.63 | 1059.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 1050.60 | 1013.54 | 1034.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 1050.60 | 1013.54 | 1034.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1050.60 | 1013.54 | 1034.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 1050.60 | 1013.54 | 1034.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1053.40 | 1013.94 | 1034.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 1008.25 | 1013.94 | 1034.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 10:15:00 | 957.84 | 1013.15 | 1033.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 09:15:00 | 907.43 | 1009.99 | 1031.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 1089.00 | 1024.40 | 1024.20 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 995.00 | 1024.22 | 1024.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 987.50 | 1019.61 | 1021.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 968.05 | 965.03 | 985.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 10:45:00 | 968.60 | 965.03 | 985.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 986.00 | 965.41 | 984.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 986.90 | 965.41 | 984.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 987.10 | 965.63 | 984.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:30:00 | 987.05 | 965.63 | 984.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 981.00 | 966.30 | 983.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:30:00 | 971.15 | 966.36 | 983.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 971.85 | 966.36 | 983.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 972.00 | 966.42 | 983.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:30:00 | 971.90 | 966.50 | 983.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 975.00 | 955.83 | 971.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 976.45 | 955.83 | 971.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 975.45 | 956.03 | 971.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 989.70 | 956.78 | 971.41 | SL hit (close>static) qty=1.00 sl=986.95 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1009.95 | 980.92 | 980.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 1016.70 | 981.76 | 981.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1042.70 | 1044.87 | 1022.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:45:00 | 1042.00 | 1044.87 | 1022.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1035.90 | 1047.99 | 1029.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1036.00 | 1047.99 | 1029.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1029.00 | 1047.67 | 1029.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1029.00 | 1047.67 | 1029.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1018.40 | 1047.38 | 1029.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 1018.40 | 1047.38 | 1029.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1019.60 | 1047.11 | 1029.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 1013.00 | 1047.11 | 1029.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1018.80 | 1046.12 | 1028.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1019.20 | 1046.12 | 1028.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1015.30 | 1045.28 | 1028.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 1015.40 | 1045.28 | 1028.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1020.30 | 1044.83 | 1028.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 1034.20 | 1044.72 | 1028.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1006.00 | 1042.65 | 1028.25 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1000.00 | 1033.15 | 1033.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 994.50 | 1032.14 | 1032.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1011.25 | 1006.76 | 1016.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1001.25 | 1006.82 | 1016.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 999.80 | 1006.60 | 1016.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1017.45 | 1005.20 | 1014.70 | SL hit (close>static) qty=1.00 sl=1016.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1063.20 | 1013.77 | 1013.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1067.20 | 1017.90 | 1015.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 1022.60 | 1023.80 | 1019.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 1022.60 | 1023.80 | 1019.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1028.70 | 1023.84 | 1019.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 1028.70 | 1023.84 | 1019.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1014.00 | 1023.87 | 1019.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1014.00 | 1023.87 | 1019.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1010.10 | 1023.73 | 1019.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1010.10 | 1023.73 | 1019.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 990.00 | 1015.45 | 1015.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 976.50 | 1015.06 | 1015.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 990.00 | 981.67 | 994.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 987.70 | 981.67 | 994.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 994.75 | 981.80 | 994.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 994.75 | 981.80 | 994.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 991.35 | 981.89 | 994.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 983.65 | 987.34 | 995.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:00:00 | 987.70 | 987.34 | 995.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 995.65 | 987.44 | 995.61 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1035.20 | 968.00 | 967.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 1050.45 | 974.64 | 971.27 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-08 12:15:00 | 671.65 | 2024-05-10 09:15:00 | 738.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-08 12:45:00 | 673.55 | 2024-05-10 09:15:00 | 740.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1008.25 | 2025-02-27 10:15:00 | 957.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1008.25 | 2025-02-28 09:15:00 | 907.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 13:15:00 | 1011.65 | 2025-03-17 09:15:00 | 1049.60 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-03-06 10:45:00 | 1038.00 | 2025-03-17 09:15:00 | 1049.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-03-06 11:15:00 | 1039.85 | 2025-03-19 09:15:00 | 1133.45 | STOP_HIT | 1.00 | -9.00% |
| SELL | retest2 | 2025-03-11 09:15:00 | 1007.95 | 2025-03-19 09:15:00 | 1133.45 | STOP_HIT | 1.00 | -12.45% |
| SELL | retest2 | 2025-03-13 09:45:00 | 1012.45 | 2025-03-19 09:15:00 | 1133.45 | STOP_HIT | 1.00 | -11.95% |
| SELL | retest2 | 2025-03-21 11:30:00 | 1013.80 | 2025-03-24 09:15:00 | 1058.35 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2025-03-21 14:00:00 | 1018.15 | 2025-03-24 09:15:00 | 1058.35 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-03-25 11:30:00 | 1009.50 | 2025-03-26 09:15:00 | 1036.35 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-26 13:15:00 | 1011.55 | 2025-04-01 09:15:00 | 960.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 13:15:00 | 1011.55 | 2025-04-07 09:15:00 | 910.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-09 10:30:00 | 971.15 | 2025-06-26 13:15:00 | 989.70 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-06-09 11:00:00 | 971.85 | 2025-06-26 13:15:00 | 989.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-06-09 12:15:00 | 972.00 | 2025-06-26 13:15:00 | 989.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-09 13:30:00 | 971.90 | 2025-06-26 13:15:00 | 989.70 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-08-25 10:00:00 | 1034.20 | 2025-08-26 10:15:00 | 1006.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-09-01 09:30:00 | 1030.20 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-09-02 14:15:00 | 1026.00 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-09-26 14:30:00 | 1029.60 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-10-30 13:30:00 | 999.80 | 2025-11-03 13:15:00 | 1017.45 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-10 11:45:00 | 999.15 | 2025-11-10 14:15:00 | 1021.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-11-10 12:15:00 | 997.00 | 2025-11-10 14:15:00 | 1021.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-24 09:15:00 | 982.10 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1006.65 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1011.20 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-12-02 11:00:00 | 1012.70 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1010.95 | 2025-12-08 15:15:00 | 1018.45 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-08 13:15:00 | 1008.80 | 2025-12-09 11:15:00 | 1026.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-09 09:30:00 | 1003.00 | 2025-12-16 09:15:00 | 960.40 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1006.85 | 2025-12-18 09:15:00 | 956.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-09 09:30:00 | 1003.00 | 2025-12-22 15:15:00 | 1005.00 | STOP_HIT | 0.50 | -0.20% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1006.85 | 2025-12-22 15:15:00 | 1005.00 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-02-13 09:15:00 | 983.65 | 2026-02-13 11:15:00 | 995.65 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-02-13 10:00:00 | 987.70 | 2026-02-13 11:15:00 | 995.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-13 14:45:00 | 987.80 | 2026-02-16 09:15:00 | 1003.15 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-13 15:15:00 | 987.00 | 2026-02-16 09:15:00 | 1003.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-24 12:15:00 | 985.65 | 2026-03-04 13:15:00 | 936.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 982.60 | 2026-03-04 13:15:00 | 933.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 985.20 | 2026-03-04 13:15:00 | 935.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 982.10 | 2026-03-04 13:15:00 | 933.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-04 13:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-04 13:15:00 | 937.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 09:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 12:15:00 | 985.65 | 2026-03-20 15:15:00 | 890.10 | TARGET_HIT | 0.50 | 9.69% |
| SELL | retest2 | 2026-02-24 13:15:00 | 982.60 | 2026-03-23 09:15:00 | 887.09 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-25 09:30:00 | 985.20 | 2026-03-23 09:15:00 | 884.34 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-02-25 12:45:00 | 982.10 | 2026-03-23 09:15:00 | 886.68 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-23 09:15:00 | 883.89 | TARGET_HIT | 0.50 | 10.63% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-23 09:15:00 | 888.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-23 09:15:00 | 874.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 11:00:00 | 985.55 | 2026-04-15 09:15:00 | 1010.00 | STOP_HIT | 1.00 | -2.48% |
