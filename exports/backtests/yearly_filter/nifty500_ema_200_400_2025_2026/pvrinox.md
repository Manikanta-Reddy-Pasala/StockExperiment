# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1075.50
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
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 0
- **Target hits / Stop hits / Partials:** 4 / 2 / 3
- **Avg / median % per leg:** 7.17% / 5.00%
- **Sum % (uncompounded):** 64.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 6 | 6 | 100.0% | 1 | 2 | 3 | 5.75% | 34.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 1 | 2 | 3 | 5.75% | 34.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 9 | 100.0% | 4 | 2 | 3 | 7.17% | 64.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 1014.60 | 984.77 | 984.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1024.45 | 986.27 | 985.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 988.50 | 989.51 | 987.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 973.65 | 989.35 | 987.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 973.65 | 989.35 | 987.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 976.75 | 989.22 | 987.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 983.00 | 989.04 | 987.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 982.05 | 988.66 | 986.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 984.90 | 988.58 | 986.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-07 09:15:00 | 1081.30 | 997.18 | 991.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1088.00 | 1106.55 | 1106.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1080.50 | 1106.13 | 1106.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1120.20 | 1095.93 | 1100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1100.10 | 1095.98 | 1100.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 1090.70 | 1096.01 | 1100.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 1036.16 | 1084.51 | 1093.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 981.63 | 1046.84 | 1067.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1066.90 | 1001.16 | 1000.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1074.20 | 1003.13 | 1001.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 09:15:00 | 983.00 | 2025-08-07 09:15:00 | 1081.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 13:00:00 | 982.05 | 2025-08-07 09:15:00 | 1080.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 14:15:00 | 984.90 | 2025-08-07 09:15:00 | 1083.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-15 13:15:00 | 1090.70 | 2025-12-23 09:15:00 | 1036.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 13:15:00 | 1090.70 | 2026-01-12 09:15:00 | 981.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1089.60 | 2026-02-16 09:15:00 | 1035.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1089.60 | 2026-02-16 09:15:00 | 1039.80 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1094.05 | 2026-02-16 09:15:00 | 1039.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1094.05 | 2026-02-16 09:15:00 | 1039.80 | STOP_HIT | 0.50 | 4.96% |
