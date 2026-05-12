# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 17
- **Target hits / Stop hits / Partials:** 2 / 17 / 1
- **Avg / median % per leg:** -0.08% / -1.11%
- **Sum % (uncompounded):** -1.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.75% | -11.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.75% | -11.9% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.56% | 10.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.56% | 10.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 3 | 15.0% | 2 | 17 | 1 | -0.08% | -1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1002.25 | 953.13 | 952.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 1017.90 | 953.77 | 953.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 989.00 | 995.99 | 979.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 989.00 | 995.99 | 979.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 977.50 | 995.67 | 979.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 974.60 | 995.67 | 979.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 989.60 | 995.61 | 979.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 996.90 | 995.53 | 979.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 15:15:00 | 977.50 | 994.64 | 979.95 | SL hit (close<static) qty=1.00 sl=977.90 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 910.05 | 991.45 | 991.65 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-22 13:15:00)

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

### Cycle 4 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1050.20 | 1034.11 | 1034.09 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1034.03 | 1034.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1033.91 | 1034.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1034.70 | 1024.53 | 1028.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1027.40 | 1024.56 | 1028.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1022.70 | 1025.96 | 1028.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 971.57 | 1022.88 | 1027.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 10:15:00 | 920.43 | 997.37 | 1012.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:00:00 | 934.95 | 2025-05-13 10:15:00 | 965.45 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-06-16 14:00:00 | 996.90 | 2025-06-17 15:15:00 | 977.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-24 11:30:00 | 998.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1003.90 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-04 14:45:00 | 997.30 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-08 13:15:00 | 987.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-08 14:00:00 | 988.50 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-28 09:15:00 | 989.50 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-01 09:15:00 | 994.80 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1020.50 | 2025-10-28 09:15:00 | 1015.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1025.90 | 2025-10-29 11:15:00 | 1128.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:00:00 | 1026.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-24 12:00:00 | 1020.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-27 13:30:00 | 1033.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1033.00 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1035.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1036.80 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-09 09:15:00 | 971.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-19 10:15:00 | 920.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 11:30:00 | 1019.80 | 2026-04-08 15:15:00 | 1035.00 | STOP_HIT | 1.00 | -1.49% |
