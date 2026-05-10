# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2025-10-27 09:15:00 → 2026-05-08 15:15:00 (917 bars)
- **Last close:** 1945.00
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
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -5.30% / -4.79%
- **Sum % (uncompounded):** -21.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.30% | -21.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.30% | -21.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.30% | -21.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1329.50 | 1431.26 | 1431.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1306.00 | 1413.90 | 1422.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1397.40 | 1385.48 | 1405.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1397.40 | 1385.48 | 1405.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1402.00 | 1385.65 | 1405.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 1389.20 | 1386.28 | 1405.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1365.00 | 1387.10 | 1404.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1387.80 | 1386.88 | 1403.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 1382.50 | 1386.53 | 1403.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1392.00 | 1386.71 | 1403.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 1398.40 | 1386.71 | 1403.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1391.60 | 1386.79 | 1402.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.99 | 1403.13 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.99 | 1403.13 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.99 | 1403.13 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.99 | 1403.13 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1503.10 | 1415.52 | 1415.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1510.10 | 1417.35 | 1416.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-09 10:30:00 | 1389.20 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1365.00 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -6.54% |
| SELL | retest2 | 2026-04-13 12:30:00 | 1387.80 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2026-04-15 11:00:00 | 1382.50 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -5.19% |
