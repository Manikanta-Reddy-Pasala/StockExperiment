# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1345.00
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 3
- **Avg / median % per leg:** -1.31% / -3.52%
- **Sum % (uncompounded):** -19.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 0 | 12 | 3 | -1.31% | -19.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 0 | 12 | 3 | -1.31% | -19.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 0 | 12 | 3 | -1.31% | -19.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1407.60 | 1485.08 | 1485.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1391.10 | 1481.20 | 1483.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 1261.00 | 1257.13 | 1329.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:30:00 | 1267.70 | 1257.13 | 1329.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1060.25 | 974.94 | 1048.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 1060.25 | 974.94 | 1048.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1033.15 | 975.52 | 1047.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1027.50 | 975.52 | 1047.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1018.15 | 978.30 | 1046.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 1075.50 | 979.27 | 1046.68 | SL hit (close>static) qty=1.00 sl=1069.45 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 12:15:00 | 1227.05 | 1071.79 | 1071.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1232.60 | 1095.51 | 1084.07 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-19 09:15:00 | 1027.50 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-03-20 09:30:00 | 1018.15 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2026-03-23 12:15:00 | 1028.05 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-23 12:45:00 | 1021.20 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-03-30 09:15:00 | 986.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-04-01 09:15:00 | 1033.00 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-04-01 13:15:00 | 1039.50 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1027.10 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2026-04-06 14:15:00 | 1026.95 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1025.35 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-07 15:00:00 | 1026.75 | 2026-04-09 09:15:00 | 1119.55 | STOP_HIT | 1.00 | -9.04% |
