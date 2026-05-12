# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 834.95
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 16 / 9
- **Target hits / Stop hits / Partials:** 5 / 12 / 8
- **Avg / median % per leg:** 1.58% / 5.00%
- **Sum % (uncompounded):** 39.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -7.00% | -35.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -7.00% | -35.0% |
| SELL (all) | 20 | 16 | 80.0% | 5 | 7 | 8 | 3.72% | 74.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 16 | 80.0% | 5 | 7 | 8 | 3.72% | 74.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 16 | 64.0% | 5 | 12 | 8 | 1.58% | 39.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 1018.90 | 1093.05 | 1093.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1008.00 | 1054.83 | 1065.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 1071.80 | 1053.09 | 1064.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1055.00 | 1053.11 | 1063.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 1047.90 | 1053.11 | 1063.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:00:00 | 1048.70 | 1052.72 | 1063.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1050.90 | 1050.46 | 1061.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 995.50 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 996.26 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 998.36 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1045.30 | 1033.84 | 1046.10 | SL hit (close>ema200) qty=0.50 sl=1033.84 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-11 09:15:00 | 1140.60 | 2025-09-08 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-08-28 14:45:00 | 1130.00 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.47% |
| BUY | retest2 | 2025-09-01 14:00:00 | 1135.40 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.91% |
| BUY | retest2 | 2025-09-02 09:15:00 | 1146.70 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -8.82% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1124.50 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1047.90 | 2025-12-09 09:15:00 | 995.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 11:00:00 | 1048.70 | 2025-12-09 09:15:00 | 996.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1050.90 | 2025-12-09 09:15:00 | 998.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1047.90 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-11-13 11:00:00 | 1048.70 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1050.90 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1043.30 | 2026-01-07 13:15:00 | 991.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1030.80 | 2026-01-08 10:15:00 | 979.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 10:15:00 | 1035.30 | 2026-01-08 10:15:00 | 983.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1034.40 | 2026-01-08 10:15:00 | 982.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1035.00 | 2026-01-08 10:15:00 | 983.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1043.30 | 2026-01-09 13:15:00 | 938.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1030.80 | 2026-01-12 09:15:00 | 927.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 10:15:00 | 1035.30 | 2026-01-12 09:15:00 | 931.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1034.40 | 2026-01-12 09:15:00 | 930.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1035.00 | 2026-01-12 09:15:00 | 931.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 12:00:00 | 845.00 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2026-04-16 15:00:00 | 850.80 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-04-20 09:30:00 | 848.40 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-04-20 15:15:00 | 850.00 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.00% |
