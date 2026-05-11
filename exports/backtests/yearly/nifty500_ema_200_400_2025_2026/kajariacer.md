# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1105.00
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
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 3
- **Avg / median % per leg:** -0.55% / -2.93%
- **Sum % (uncompounded):** -8.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.42% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.42% | -20.5% |
| SELL (all) | 9 | 6 | 66.7% | 0 | 6 | 3 | 1.36% | 12.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 6 | 66.7% | 0 | 6 | 3 | 1.36% | 12.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 0 | 12 | 3 | -0.55% | -8.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1047.65 | 894.90 | 894.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1050.95 | 916.59 | 905.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 11:15:00 | 1223.90 | 1228.54 | 1168.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 12:00:00 | 1223.90 | 1228.54 | 1168.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1204.00 | 1229.13 | 1195.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 1221.00 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1221.90 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1182.60 | 1221.90 | 1196.43 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1115.00 | 1198.35 | 1198.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1106.60 | 1188.43 | 1193.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1092.50 | 1091.89 | 1127.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1093.00 | 1091.89 | 1127.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1007.60 | 952.62 | 991.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1001.30 | 952.62 | 991.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1003.55 | 953.13 | 991.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:30:00 | 1000.05 | 954.11 | 992.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 998.80 | 960.26 | 991.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 996.00 | 960.64 | 991.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 950.05 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 948.86 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 946.20 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 962.50 | 961.98 | 989.76 | SL hit (close>ema200) qty=0.50 sl=961.98 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1115.05 | 974.83 | 974.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 1127.40 | 981.76 | 978.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 13:15:00 | 1107.50 | 1109.44 | 1057.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 14:00:00 | 1107.50 | 1109.44 | 1057.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1056.80 | 1107.53 | 1057.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 1055.80 | 1107.53 | 1057.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1065.70 | 1107.11 | 1058.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 1055.10 | 1107.11 | 1058.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:15:00 | 855.85 | 2025-05-14 10:15:00 | 883.85 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-05-14 09:30:00 | 854.95 | 2025-05-14 11:15:00 | 890.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-09-24 12:45:00 | 1221.00 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-09-24 13:15:00 | 1221.90 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-10-10 09:45:00 | 1234.20 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1220.70 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-10-24 14:15:00 | 1211.00 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-10-27 11:00:00 | 1208.90 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-02-18 12:30:00 | 1000.05 | 2026-02-24 14:15:00 | 950.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 998.80 | 2026-02-24 14:15:00 | 948.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:30:00 | 996.00 | 2026-02-24 14:15:00 | 946.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 12:30:00 | 1000.05 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2026-02-23 09:30:00 | 998.80 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2026-02-23 10:30:00 | 996.00 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2026-04-07 14:15:00 | 1000.10 | 2026-04-08 09:15:00 | 1061.25 | STOP_HIT | 1.00 | -6.11% |
