# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1013.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 1.56% / 0.77%
- **Sum % (uncompounded):** 17.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.16% | -10.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.16% | -10.8% |
| SELL (all) | 6 | 6 | 100.0% | 1 | 2 | 3 | 4.66% | 27.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 1 | 2 | 3 | 4.66% | 27.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 6 | 54.5% | 1 | 7 | 3 | 1.56% | 17.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 1167.90 | 1215.89 | 1216.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1164.20 | 1214.93 | 1215.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1155.00 | 1152.79 | 1174.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:30:00 | 1155.90 | 1152.79 | 1174.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1166.10 | 1153.05 | 1171.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1168.10 | 1153.05 | 1171.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1172.10 | 1154.00 | 1171.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1172.10 | 1154.00 | 1171.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1176.10 | 1154.22 | 1171.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1176.10 | 1154.22 | 1171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1175.00 | 1154.42 | 1171.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1184.80 | 1154.42 | 1171.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1183.30 | 1167.28 | 1175.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 1183.30 | 1167.28 | 1175.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1180.80 | 1168.43 | 1175.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 1180.80 | 1168.43 | 1175.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1179.20 | 1168.54 | 1175.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 1182.40 | 1168.54 | 1175.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1179.00 | 1168.65 | 1175.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1179.00 | 1168.65 | 1175.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1187.90 | 1168.84 | 1175.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1186.20 | 1168.84 | 1175.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1180.80 | 1170.34 | 1176.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 1180.80 | 1170.34 | 1176.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1174.00 | 1170.38 | 1176.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1173.20 | 1170.44 | 1176.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1156.70 | 1170.69 | 1176.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:15:00 | 1114.54 | 1159.49 | 1169.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 14:15:00 | 1098.87 | 1158.86 | 1168.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1147.80 | 1140.31 | 1156.48 | SL hit (close>ema200) qty=0.50 sl=1140.31 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-18 09:30:00 | 1225.60 | 2025-08-18 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-18 10:45:00 | 1224.80 | 2025-08-18 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-19 13:30:00 | 1224.90 | 2025-08-25 09:15:00 | 1215.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-20 09:45:00 | 1226.90 | 2025-08-26 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1233.90 | 2025-08-26 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-10-30 12:30:00 | 1173.20 | 2025-11-10 13:15:00 | 1114.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1156.70 | 2025-11-10 14:15:00 | 1098.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 12:30:00 | 1173.20 | 2025-11-19 10:15:00 | 1147.80 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1156.70 | 2025-11-19 10:15:00 | 1147.80 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-11-24 10:15:00 | 1172.00 | 2025-12-02 10:15:00 | 1113.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 10:15:00 | 1172.00 | 2025-12-17 10:15:00 | 1054.80 | TARGET_HIT | 0.50 | 10.00% |
