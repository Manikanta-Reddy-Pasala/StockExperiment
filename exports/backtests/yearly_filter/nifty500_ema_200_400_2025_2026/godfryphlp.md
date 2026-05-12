# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2424.80
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
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -3.32% / -2.89%
- **Sum % (uncompounded):** -36.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.33% | -16.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.33% | -16.3% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.06% | -20.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.06% | -20.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.32% | -36.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 3103.80 | 3283.85 | 3284.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 3028.00 | 3276.20 | 3280.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 2206.60 | 2200.20 | 2441.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:45:00 | 2212.20 | 2200.20 | 2441.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2386.80 | 2170.85 | 2365.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2386.80 | 2170.85 | 2365.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2382.50 | 2172.96 | 2365.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:30:00 | 2388.30 | 2172.96 | 2365.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2478.90 | 2176.00 | 2366.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2478.90 | 2176.00 | 2366.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2112.30 | 2024.65 | 2128.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 2118.40 | 2024.65 | 2128.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 2130.70 | 2026.48 | 2128.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:00:00 | 2130.70 | 2026.48 | 2128.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 2106.30 | 2027.27 | 2128.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 2090.60 | 2027.27 | 2128.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2094.70 | 2029.85 | 2127.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 2230.90 | 2037.18 | 2127.76 | SL hit (close>static) qty=1.00 sl=2146.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-29 13:00:00 | 3366.00 | 2025-10-09 15:15:00 | 3295.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-10-01 13:30:00 | 3336.90 | 2025-10-09 15:15:00 | 3295.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-10 09:30:00 | 3343.00 | 2025-10-13 09:15:00 | 3292.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-10 11:00:00 | 3330.50 | 2025-10-13 09:15:00 | 3292.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-14 09:15:00 | 3349.80 | 2025-10-14 10:15:00 | 3305.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-15 09:15:00 | 3368.10 | 2025-10-23 09:15:00 | 3231.80 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-10-17 09:15:00 | 3398.10 | 2025-10-23 09:15:00 | 3231.80 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-04-15 13:15:00 | 2090.60 | 2026-04-17 09:15:00 | 2230.90 | STOP_HIT | 1.00 | -6.71% |
| SELL | retest2 | 2026-04-16 09:30:00 | 2094.70 | 2026-04-17 09:15:00 | 2230.90 | STOP_HIT | 1.00 | -6.50% |
| SELL | retest2 | 2026-04-22 09:15:00 | 2102.40 | 2026-04-22 09:15:00 | 2163.20 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-04-24 13:30:00 | 2088.40 | 2026-04-29 10:15:00 | 2174.90 | STOP_HIT | 1.00 | -4.14% |
