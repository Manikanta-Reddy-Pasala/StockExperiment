# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3098.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 4 / 1 / 3
- **Avg / median % per leg:** 6.70% / 10.00%
- **Sum % (uncompounded):** 53.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.23% | 43.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.23% | 43.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 7 | 87.5% | 4 | 1 | 3 | 6.70% | 53.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 1198.80 | 1225.69 | 1225.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1116.40 | 1223.32 | 1224.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1221.01 | 1175.17 | 1194.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1221.01 | 1175.17 | 1194.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1221.01 | 1175.17 | 1194.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 1219.40 | 1175.17 | 1194.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1219.00 | 1175.61 | 1194.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 14:00:00 | 1210.99 | 1176.81 | 1195.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 1228.00 | 1178.18 | 1195.72 | SL hit (close>static) qty=1.00 sl=1224.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 1245.00 | 1096.68 | 1096.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1248.30 | 1114.47 | 1105.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1137.30 | 1159.07 | 1132.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1137.30 | 1159.07 | 1132.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1137.30 | 1159.07 | 1132.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 1133.30 | 1159.07 | 1132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 1129.70 | 1158.57 | 1132.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:00:00 | 1129.70 | 1158.57 | 1132.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1124.80 | 1158.23 | 1132.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:00:00 | 1124.80 | 1158.23 | 1132.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1137.80 | 1157.43 | 1132.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 1193.50 | 1157.43 | 1132.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 10:15:00 | 1312.85 | 1226.77 | 1183.38 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-05 14:00:00 | 1210.99 | 2025-02-06 09:15:00 | 1228.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-02-06 13:45:00 | 1208.86 | 2025-02-11 09:15:00 | 1148.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1207.60 | 2025-02-11 09:15:00 | 1147.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:30:00 | 1209.75 | 2025-02-11 09:15:00 | 1149.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 13:45:00 | 1208.86 | 2025-02-11 12:15:00 | 1087.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1207.60 | 2025-02-11 12:15:00 | 1086.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 13:30:00 | 1209.75 | 2025-02-11 12:15:00 | 1088.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1193.50 | 2025-05-29 10:15:00 | 1312.85 | TARGET_HIT | 1.00 | 10.00% |
