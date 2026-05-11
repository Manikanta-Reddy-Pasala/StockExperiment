# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 2 / 9 / 2
- **Avg / median % per leg:** 1.75% / -0.52%
- **Sum % (uncompounded):** 22.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 1 | 7 | 0 | 0.36% | 2.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 1 | 7 | 0 | 0.36% | 2.9% |
| SELL (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.97% | 19.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.97% | 19.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 5 | 38.5% | 2 | 9 | 2 | 1.75% | 22.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 932.60 | 982.89 | 982.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 923.90 | 981.75 | 982.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 973.20 | 972.26 | 977.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 973.20 | 972.26 | 977.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 977.05 | 972.30 | 977.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 979.75 | 972.30 | 977.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 981.75 | 972.39 | 977.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 980.70 | 972.39 | 977.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 977.65 | 972.44 | 977.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 974.25 | 972.44 | 977.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 977.40 | 972.49 | 977.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 981.00 | 972.49 | 977.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 976.05 | 972.53 | 977.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:15:00 | 978.70 | 972.53 | 977.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 975.85 | 972.56 | 977.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 973.80 | 972.89 | 977.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 925.11 | 968.84 | 974.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 960.95 | 952.19 | 963.30 | SL hit (close>ema200) qty=0.50 sl=952.19 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1109.00 | 971.70 | 971.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 1128.40 | 973.26 | 972.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1030.03 | 1009.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 1027.80 | 1030.03 | 1009.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.45 | 1016.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1019.70 | 1034.45 | 1016.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1017.90 | 1033.86 | 1016.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1015.70 | 1033.86 | 1016.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.69 | 1016.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1020.50 | 1033.69 | 1016.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1015.40 | 1033.51 | 1016.34 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 1013.40 | 1033.43 | 1033.52 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 1045.80 | 1033.64 | 1033.62 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1022.10 | 1033.63 | 1033.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1020.40 | 1033.50 | 1033.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1034.70 | 1024.53 | 1028.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1027.40 | 1024.56 | 1028.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1022.70 | 1025.96 | 1028.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 971.57 | 1022.88 | 1026.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 10:15:00 | 920.43 | 997.37 | 1012.39 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.56 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 13:15:00 | 973.80 | 2025-08-28 09:15:00 | 925.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:15:00 | 973.80 | 2025-09-10 09:15:00 | 960.95 | STOP_HIT | 0.50 | 1.32% |
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
