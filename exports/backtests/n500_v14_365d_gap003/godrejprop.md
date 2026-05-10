# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1874.80
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
- **Avg / median % per leg:** -2.56% / -2.44%
- **Sum % (uncompounded):** -10.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.56% | -10.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.56% | -10.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.56% | -10.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 2313.50 | 2097.01 | 2096.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2329.00 | 2118.83 | 2107.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 2190.50 | 2192.59 | 2152.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 15:00:00 | 2190.50 | 2192.59 | 2152.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 2144.50 | 2192.12 | 2152.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 2154.40 | 2192.12 | 2152.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 2183.80 | 2192.04 | 2152.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 2190.90 | 2185.86 | 2152.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:00:00 | 2188.30 | 2185.67 | 2153.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 2195.00 | 2187.97 | 2156.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 2200.50 | 2188.19 | 2157.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2156.30 | 2188.01 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2156.30 | 2188.01 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2159.10 | 2187.72 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 2154.80 | 2187.72 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 2165.00 | 2187.50 | 2158.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-12-03 12:15:00 | 2073.00 | 2139.59 | 2139.72 | min_gap filter: gap=0.006% < 0.030% |
| TREND_RESET | 2025-12-03 12:15:00 | 2073.00 | 2139.59 | 2139.72 | EMA inversion without crossover edge (EMA200=2139.59 EMA400=2139.72) — end cycle |
| CROSSOVER_SKIP | 2026-05-04 13:15:00 | 1876.50 | 1740.27 | 1740.11 | min_gap filter: gap=0.009% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-12 09:15:00 | 2190.90 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-11-12 13:00:00 | 2188.30 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-11-14 10:45:00 | 2195.00 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-11-14 15:00:00 | 2200.50 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.87% |
