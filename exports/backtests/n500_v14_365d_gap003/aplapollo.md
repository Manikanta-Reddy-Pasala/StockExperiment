# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1950.00
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
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.41% / -1.38%
- **Sum % (uncompounded):** -9.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.41% | -9.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.41% | -9.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.41% | -9.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1551.60 | 1716.09 | 1716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1497.50 | 1713.92 | 1715.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1638.10 | 1630.49 | 1660.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:45:00 | 1641.20 | 1630.49 | 1660.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1665.00 | 1630.71 | 1659.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1665.00 | 1630.71 | 1659.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1664.90 | 1631.05 | 1659.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1659.90 | 1631.30 | 1659.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 1659.10 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1659.30 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1655.00 | 1631.89 | 1653.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1655.00 | 1632.58 | 1653.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1674.10 | 1632.58 | 1653.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1668.00 | 1632.94 | 1653.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 1647.20 | 1633.14 | 1653.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 1653.30 | 1638.01 | 1654.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 1653.60 | 1638.17 | 1654.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |
| CROSSOVER_SKIP | 2025-09-23 09:15:00 | 1669.70 | 1666.01 | 1666.00 | min_gap filter: gap=0.001% < 0.030% |
| TREND_RESET | 2025-09-23 09:15:00 | 1669.70 | 1666.01 | 1666.00 | EMA inversion without crossover edge (EMA200=1666.01 EMA400=1666.00) — end cycle |
| CROSSOVER_SKIP | 2026-04-06 12:15:00 | 1910.30 | 2018.31 | 2018.83 | min_gap filter: gap=0.027% < 0.030% |
| CROSSOVER_SKIP | 2026-04-17 13:15:00 | 2106.90 | 2017.77 | 2017.58 | min_gap filter: gap=0.009% < 0.030% |
| CROSSOVER_SKIP | 2026-04-30 13:15:00 | 1902.30 | 2021.13 | 2021.23 | min_gap filter: gap=0.005% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 12:30:00 | 1659.90 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-02 11:30:00 | 1659.10 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1659.30 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1655.00 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-03 11:15:00 | 1647.20 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1653.30 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-05 13:00:00 | 1653.60 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.59% |
