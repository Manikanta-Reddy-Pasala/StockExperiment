# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 903.00
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
| ENTRY2 | 5 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 0
- **Target hits / Stop hits / Partials:** 5 / 0 / 5
- **Avg / median % per leg:** 7.50% / 9.75%
- **Sum % (uncompounded):** 75.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1113.60 | 1217.56 | 1217.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1103.80 | 1190.84 | 1203.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 1124.30 | 1123.81 | 1151.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 12:00:00 | 1124.30 | 1123.81 | 1151.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1153.10 | 1124.92 | 1148.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1153.10 | 1124.92 | 1148.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1139.80 | 1125.07 | 1148.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 1136.70 | 1125.07 | 1148.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 1134.00 | 1125.16 | 1148.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 1137.10 | 1125.70 | 1148.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 1138.80 | 1126.35 | 1147.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1139.10 | 1127.34 | 1147.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:45:00 | 1154.30 | 1127.34 | 1147.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1150.00 | 1127.57 | 1147.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1151.90 | 1127.78 | 1147.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1145.90 | 1127.96 | 1147.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 1138.30 | 1128.06 | 1147.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1079.87 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1077.30 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1080.24 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1081.86 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1081.38 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-07 10:15:00 | 1023.03 | 1096.92 | 1123.80 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-07 10:15:00 | 1023.39 | 1096.92 | 1123.80 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-07 10:15:00 | 1024.92 | 1096.92 | 1123.80 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-07 10:15:00 | 1024.47 | 1096.92 | 1123.80 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-07 11:15:00 | 1020.60 | 1096.21 | 1123.31 | Target hit (10%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-12-02 14:15:00 | 1126.70 | 1099.28 | 1099.18 | min_gap filter: gap=0.009% < 0.030% |
| TREND_RESET | 2025-12-02 14:15:00 | 1126.70 | 1099.28 | 1099.18 | EMA inversion without crossover edge (EMA200=1099.28 EMA400=1099.18) — end cycle |
| CROSSOVER_SKIP | 2026-01-09 13:15:00 | 999.00 | 1108.17 | 1108.40 | min_gap filter: gap=0.023% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-16 12:15:00 | 1136.70 | 2025-09-25 09:15:00 | 1079.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 13:00:00 | 1134.00 | 2025-09-25 09:15:00 | 1077.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 1137.10 | 2025-09-25 09:15:00 | 1080.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1138.80 | 2025-09-25 09:15:00 | 1081.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:00:00 | 1138.30 | 2025-09-25 09:15:00 | 1081.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:15:00 | 1136.70 | 2025-10-07 10:15:00 | 1023.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 13:00:00 | 1134.00 | 2025-10-07 10:15:00 | 1023.39 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2025-09-17 09:45:00 | 1137.10 | 2025-10-07 10:15:00 | 1024.92 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1138.80 | 2025-10-07 10:15:00 | 1024.47 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2025-09-19 12:00:00 | 1138.30 | 2025-10-07 11:15:00 | 1020.60 | TARGET_HIT | 0.50 | 10.34% |
