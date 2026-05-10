# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1105.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** -3.42% / -3.14%
- **Sum % (uncompounded):** -20.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.42% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.42% | -20.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.42% | -20.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1047.65 | 894.90 | 894.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1050.95 | 916.59 | 905.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 11:15:00 | 1223.90 | 1228.54 | 1168.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 12:00:00 | 1223.90 | 1228.54 | 1168.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1204.00 | 1229.13 | 1195.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 1221.00 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1221.90 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1182.60 | 1221.90 | 1196.43 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1182.60 | 1221.90 | 1196.43 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 1234.20 | 1210.06 | 1196.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1220.70 | 1216.57 | 1200.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1207.70 | 1222.36 | 1206.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:15:00 | 1204.20 | 1222.36 | 1206.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1204.50 | 1222.19 | 1206.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1204.40 | 1222.19 | 1206.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1200.00 | 1221.96 | 1206.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1200.00 | 1221.96 | 1206.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1206.40 | 1221.81 | 1206.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 1211.00 | 1221.81 | 1206.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 1208.90 | 1221.28 | 1206.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1175.50 | 1219.56 | 1208.15 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1175.50 | 1219.56 | 1208.15 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1175.50 | 1219.56 | 1208.15 | SL hit (close<static) qty=1.00 sl=1198.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1175.50 | 1219.56 | 1208.15 | SL hit (close<static) qty=1.00 sl=1198.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-11 09:15:00 | 1115.00 | 1198.35 | 1198.63 | min_gap filter: gap=0.025% < 0.030% |
| TREND_RESET | 2025-11-11 09:15:00 | 1115.00 | 1198.35 | 1198.63 | EMA inversion without crossover edge (EMA200=1198.35 EMA400=1198.63) — end cycle |
| CROSSOVER_SKIP | 2026-04-10 15:15:00 | 1115.05 | 974.83 | 974.67 | min_gap filter: gap=0.014% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-24 12:45:00 | 1221.00 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-09-24 13:15:00 | 1221.90 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-10-10 09:45:00 | 1234.20 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1220.70 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-10-24 14:15:00 | 1211.00 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-10-27 11:00:00 | 1208.90 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.76% |
