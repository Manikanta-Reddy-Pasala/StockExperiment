# AXISBANK (AXISBANK)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1270.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 0
- **Avg / median % per leg:** 0.09% / -0.62%
- **Sum % (uncompounded):** 0.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.09% | 0.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.09% | 0.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.09% | 0.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:45:00 | 1252.10 | 1249.64 | 1270.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 1316.00 | 1250.28 | 1270.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 15:15:00 | 1364.00 | 1287.07 | 1286.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1311.89 | 1300.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1301.30 | 1311.89 | 1300.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1294.70 | 1311.73 | 1300.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1291.60 | 1311.53 | 1300.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1291.30 | 1311.53 | 1300.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1296.60 | 1310.11 | 1300.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1296.60 | 1310.11 | 1300.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1296.00 | 1309.97 | 1300.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 1290.60 | 1309.97 | 1300.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1290.60 | 1309.78 | 1300.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1265.90 | 1309.78 | 1300.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1300.60 | 1300.04 | 1296.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1299.70 | 1300.04 | 1296.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1298.50 | 1300.02 | 1296.19 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-18 10:30:00 | 1235.90 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-18 12:00:00 | 1235.50 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-18 12:30:00 | 1235.60 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1240.20 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-22 14:30:00 | 1233.50 | 2025-12-23 14:15:00 | 1225.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-22 15:15:00 | 1233.80 | 2025-12-23 14:15:00 | 1225.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-24 11:15:00 | 1234.60 | 2025-12-24 12:15:00 | 1228.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1235.20 | 2026-01-28 09:15:00 | 1358.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 09:15:00 | 1305.30 | 2026-03-12 09:15:00 | 1245.80 | STOP_HIT | 1.00 | -4.56% |
