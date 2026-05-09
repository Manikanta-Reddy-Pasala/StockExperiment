# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 2502.00
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
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 1
- **Avg / median % per leg:** -0.02% / -1.70%
- **Sum % (uncompounded):** -0.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.02% | -0.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.02% | -0.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.02% | -0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.30 | 2396.43 | 2397.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2394.97 | 2396.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 2272.00 | 2270.61 | 2310.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 2277.90 | 2270.61 | 2310.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 2180.00 | 2117.03 | 2195.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2208.10 | 2117.93 | 2195.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 2215.30 | 2117.93 | 2195.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2215.90 | 2118.91 | 2195.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 2226.60 | 2118.91 | 2195.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2206.50 | 2122.42 | 2195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2205.00 | 2122.42 | 2195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2198.50 | 2124.13 | 2195.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 2168.60 | 2162.97 | 2202.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2191.10 | 2162.90 | 2199.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.67 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.67 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:30:00 | 2190.50 | 2166.92 | 2200.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 2195.30 | 2169.22 | 2200.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2192.20 | 2168.53 | 2197.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2198.60 | 2168.53 | 2197.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2188.90 | 2168.73 | 2197.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 2180.00 | 2168.85 | 2197.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 2180.60 | 2169.02 | 2197.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 2178.10 | 2169.97 | 2197.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 2200.90 | 2170.64 | 2197.09 | SL hit (close>static) qty=1.00 sl=2200.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 2200.90 | 2170.64 | 2197.09 | SL hit (close>static) qty=1.00 sl=2200.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 2200.90 | 2170.64 | 2197.09 | SL hit (close>static) qty=1.00 sl=2200.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2230.70 | 2172.48 | 2197.36 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2230.70 | 2172.48 | 2197.36 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 2160.20 | 2176.44 | 2197.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 2052.19 | 2170.50 | 2193.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 1944.18 | 2149.27 | 2180.32 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 2052.00 | 1961.37 | 2045.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:45:00 | 2062.00 | 1961.37 | 2045.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 2035.70 | 1962.11 | 2045.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:45:00 | 2030.10 | 1962.85 | 2045.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 2021.00 | 1965.81 | 2045.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 2064.70 | 1972.17 | 2045.35 | SL hit (close>static) qty=1.00 sl=2060.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 2064.70 | 1972.17 | 2045.35 | SL hit (close>static) qty=1.00 sl=2060.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-13 09:15:00 | 2168.60 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-02-17 10:30:00 | 2191.10 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-18 10:30:00 | 2190.50 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-02-19 09:45:00 | 2195.30 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-25 14:15:00 | 2230.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-02-24 12:00:00 | 2178.10 | 2026-02-25 14:15:00 | 2230.70 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2160.20 | 2026-03-04 09:15:00 | 2052.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2160.20 | 2026-03-09 09:15:00 | 1944.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 12:45:00 | 2030.10 | 2026-04-10 10:15:00 | 2064.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-09 09:30:00 | 2021.00 | 2026-04-10 10:15:00 | 2064.70 | STOP_HIT | 1.00 | -2.16% |
