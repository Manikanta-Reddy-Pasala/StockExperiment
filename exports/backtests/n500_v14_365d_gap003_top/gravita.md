# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1760.60
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
| ALERT3 | 1 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 3.11% / 2.36%
- **Sum % (uncompounded):** 9.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.11% | 9.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.11% | 9.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.11% | 9.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1561.50 | 1754.53 | 1755.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1547.80 | 1737.27 | 1746.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 1674.90 | 1651.42 | 1693.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:00:00 | 1674.90 | 1651.42 | 1693.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1719.20 | 1653.66 | 1687.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1712.00 | 1654.26 | 1687.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1626.40 | 1659.17 | 1686.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 1671.60 | 1658.72 | 1685.47 | SL hit (close>ema200) qty=0.50 sl=1658.72 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 1707.80 | 1558.47 | 1561.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| CROSSOVER_SKIP | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | min_gap filter: gap=0.008% < 0.030% |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | Force close (TREND_INVERSION) qty=1.00 alert=retest2 |
| TREND_RESET | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | EMA inversion without crossover edge (EMA200=1565.12 EMA400=1564.99) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-10 10:45:00 | 1712.00 | 2026-02-16 09:15:00 | 1626.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 10:45:00 | 1712.00 | 2026-02-16 14:15:00 | 1671.60 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2026-05-04 10:00:00 | 1707.80 | 2026-05-04 15:15:00 | 1674.00 | STOP_HIT | 1.00 | 1.98% |
