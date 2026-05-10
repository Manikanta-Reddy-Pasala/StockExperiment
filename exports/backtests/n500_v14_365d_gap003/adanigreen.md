# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 0
- **Avg / median % per leg:** 0.36% / -1.09%
- **Sum % (uncompounded):** 2.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 1 | 7 | 0 | 0.36% | 2.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 1 | 7 | 0 | 0.36% | 2.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 1 | 12.5% | 1 | 7 | 0 | 0.36% | 2.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1148.65 | 975.00 | 974.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 1158.00 | 976.82 | 975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1030.03 | 1010.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 1027.80 | 1030.03 | 1010.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.45 | 1017.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1019.70 | 1034.45 | 1017.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1017.20 | 1034.02 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1017.20 | 1034.02 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1017.90 | 1033.86 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1015.70 | 1033.86 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.69 | 1017.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1020.50 | 1033.69 | 1017.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1015.40 | 1033.51 | 1017.10 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1025.90 | 1031.97 | 1016.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-29 11:15:00 | 1128.49 | 1033.55 | 1017.82 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:00:00 | 1026.00 | 1061.56 | 1044.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 1020.00 | 1060.74 | 1043.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1014.70 | 1059.87 | 1043.49 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1014.70 | 1059.87 | 1043.49 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1029.10 | 1053.40 | 1041.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 1033.20 | 1052.71 | 1041.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1033.00 | 1052.49 | 1041.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 1035.20 | 1052.18 | 1041.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1036.80 | 1051.25 | 1041.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1039.00 | 1051.13 | 1041.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1040.50 | 1051.13 | 1041.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1039.60 | 1051.01 | 1041.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1036.20 | 1051.01 | 1041.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1042.30 | 1050.93 | 1041.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1035.10 | 1050.93 | 1041.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1028.40 | 1050.70 | 1041.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1028.40 | 1050.70 | 1041.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1021.70 | 1049.70 | 1040.90 | SL hit (close<static) qty=1.00 sl=1024.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1021.70 | 1049.70 | 1040.90 | SL hit (close<static) qty=1.00 sl=1024.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1021.70 | 1049.70 | 1040.90 | SL hit (close<static) qty=1.00 sl=1024.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1021.70 | 1049.70 | 1040.90 | SL hit (close<static) qty=1.00 sl=1024.20 alert=retest2 |
| CROSSOVER_SKIP | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | min_gap filter: gap=0.000% < 0.030% |
| TREND_RESET | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | EMA inversion without crossover edge (EMA200=1034.12 EMA400=1034.12) — end cycle |
| CROSSOVER_SKIP | 2025-12-15 12:15:00 | 1050.20 | 1034.11 | 1034.09 | min_gap filter: gap=0.002% < 0.030% |
| CROSSOVER_SKIP | 2025-12-16 15:15:00 | 1024.00 | 1034.03 | 1034.06 | min_gap filter: gap=0.003% < 0.030% |

### Cycle 2 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-28 09:15:00 | 1020.50 | 2025-10-28 09:15:00 | 1015.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1025.90 | 2025-10-29 11:15:00 | 1128.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:00:00 | 1026.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-24 12:00:00 | 1020.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-27 13:30:00 | 1033.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1033.00 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1035.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1036.80 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.46% |
