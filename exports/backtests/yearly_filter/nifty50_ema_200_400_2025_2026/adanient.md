# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 1
- **Avg / median % per leg:** 0.94% / -0.96%
- **Sum % (uncompounded):** 15.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 15 | 2 | 13.3% | 1 | 13 | 1 | -0.27% | -4.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 1 | 13 | 1 | -0.27% | -4.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 4 | 23.5% | 3 | 13 | 1 | 0.94% | 15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2196.08 | 2430.63 | 2430.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2172.13 | 2428.06 | 2429.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2308.54 | 2272.40 | 2320.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 2308.54 | 2272.40 | 2320.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2333.36 | 2273.01 | 2320.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 2334.42 | 2273.01 | 2320.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 2336.56 | 2273.64 | 2320.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 2317.84 | 2274.00 | 2320.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2342.86 | 2276.18 | 2320.89 | SL hit (close>static) qty=1.00 sl=2341.21 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2536.66 | 2348.59 | 2348.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.30 | 2396.43 | 2397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2394.97 | 2396.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 2272.00 | 2270.61 | 2310.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 2277.90 | 2270.61 | 2310.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 2180.00 | 2117.03 | 2195.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2208.10 | 2117.93 | 2195.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 2215.30 | 2117.93 | 2195.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2215.90 | 2118.91 | 2195.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 2226.60 | 2118.91 | 2195.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2206.50 | 2122.42 | 2195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2205.00 | 2122.42 | 2195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2198.50 | 2124.13 | 2195.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 2168.60 | 2162.97 | 2202.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2191.10 | 2162.90 | 2199.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.66 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 2309.89 | 2025-06-10 11:15:00 | 2540.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:45:00 | 2328.70 | 2025-06-10 11:15:00 | 2561.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-11 12:30:00 | 2317.84 | 2025-09-12 09:15:00 | 2342.86 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-12 11:30:00 | 2329.67 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-17 11:15:00 | 2331.90 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-17 12:00:00 | 2330.35 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.93% |
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
