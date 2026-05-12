# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 917.00
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
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 0
- **Target hits / Stop hits / Partials:** 4 / 2 / 2
- **Avg / median % per leg:** 6.42% / 10.00%
- **Sum % (uncompounded):** 51.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.84% | 11.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.84% | 11.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 8 | 100.0% | 4 | 2 | 2 | 6.42% | 51.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1093.00 | 1193.90 | 1194.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1081.00 | 1192.78 | 1193.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1116.10 | 1110.36 | 1138.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 1116.10 | 1110.36 | 1138.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1138.70 | 1109.95 | 1132.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1138.70 | 1109.95 | 1132.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1144.50 | 1110.30 | 1132.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 1143.50 | 1110.30 | 1132.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1141.40 | 1111.50 | 1132.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 1141.40 | 1111.50 | 1132.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1135.00 | 1114.78 | 1133.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 1135.20 | 1114.78 | 1133.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1142.80 | 1115.06 | 1133.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 1142.80 | 1115.06 | 1133.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1112.00 | 1083.05 | 1107.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1112.50 | 1083.05 | 1107.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1121.50 | 1083.43 | 1107.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1121.50 | 1083.43 | 1107.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1109.50 | 1088.32 | 1108.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1110.70 | 1088.32 | 1108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1120.60 | 1088.95 | 1108.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1121.00 | 1088.95 | 1108.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1118.20 | 1112.91 | 1116.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 1116.40 | 1113.57 | 1117.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 1060.58 | 1109.05 | 1114.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 1112.20 | 1105.06 | 1111.98 | SL hit (close>ema200) qty=0.50 sl=1105.06 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-03 09:15:00 | 1160.80 | 2025-06-10 10:15:00 | 1276.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-03 10:15:00 | 1160.40 | 2025-06-10 10:15:00 | 1276.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1161.20 | 2025-06-10 10:15:00 | 1277.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 11:15:00 | 1162.30 | 2025-06-10 10:15:00 | 1278.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-30 10:00:00 | 1116.40 | 2025-11-06 09:15:00 | 1060.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 10:00:00 | 1116.40 | 2025-11-07 14:15:00 | 1112.20 | STOP_HIT | 0.50 | 0.38% |
| SELL | retest2 | 2025-11-12 10:00:00 | 1113.00 | 2025-11-24 14:15:00 | 1057.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:00:00 | 1113.00 | 2025-12-01 10:15:00 | 1102.10 | STOP_HIT | 0.50 | 0.98% |
