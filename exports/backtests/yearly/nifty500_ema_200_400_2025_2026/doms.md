# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2340.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 19 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 1 / 20 / 3
- **Avg / median % per leg:** -0.01% / -1.07%
- **Sum % (uncompounded):** -0.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 9 | 37.5% | 1 | 20 | 3 | -0.01% | -0.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.71% | -8.1% |
| SELL @ 3rd Alert (retest2) | 21 | 9 | 42.9% | 1 | 17 | 3 | 0.37% | 7.8% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.71% | -8.1% |
| retest2 (combined) | 21 | 9 | 42.9% | 1 | 17 | 3 | 0.37% | 7.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 2526.80 | 2727.23 | 2727.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2490.00 | 2718.89 | 2723.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 2486.40 | 2483.48 | 2563.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:00:00 | 2486.40 | 2483.48 | 2563.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2446.40 | 2381.72 | 2438.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 2446.40 | 2381.72 | 2438.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2517.00 | 2383.07 | 2439.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 2517.00 | 2383.07 | 2439.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2545.10 | 2384.68 | 2439.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 2545.10 | 2384.68 | 2439.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2513.20 | 2396.67 | 2439.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2505.90 | 2396.67 | 2439.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2500.40 | 2397.70 | 2440.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 2488.40 | 2398.59 | 2440.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 2485.00 | 2401.46 | 2441.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 2477.20 | 2417.93 | 2445.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 2495.60 | 2424.14 | 2447.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2460.50 | 2427.58 | 2448.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 2455.00 | 2427.88 | 2448.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 2459.70 | 2428.57 | 2448.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 2458.70 | 2428.87 | 2448.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 2435.70 | 2429.24 | 2448.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2448.30 | 2429.60 | 2448.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 2448.30 | 2429.60 | 2448.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 2451.30 | 2429.95 | 2448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 2451.30 | 2429.95 | 2448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2443.70 | 2430.09 | 2448.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 2436.10 | 2430.24 | 2448.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 2436.20 | 2430.47 | 2448.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:00:00 | 2440.60 | 2430.58 | 2448.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:30:00 | 2441.50 | 2430.97 | 2447.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 2446.80 | 2431.13 | 2447.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 2451.80 | 2431.13 | 2447.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2486.10 | 2431.67 | 2448.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 2486.10 | 2431.67 | 2448.04 | SL hit (close>static) qty=1.00 sl=2471.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 2661.50 | 2463.52 | 2462.78 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 2390.50 | 2544.20 | 2544.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 2351.70 | 2521.40 | 2532.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 2468.90 | 2462.08 | 2495.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2425.60 | 2461.91 | 2495.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 13:15:00 | 2434.90 | 2460.60 | 2494.15 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 15:15:00 | 2420.20 | 2460.13 | 2493.58 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2420.20 | 2459.73 | 2493.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 2492.70 | 2455.54 | 2488.10 | SL hit (close>ema400) qty=1.00 sl=2488.10 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 11:45:00 | 2488.40 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-08-18 15:00:00 | 2485.00 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-22 09:15:00 | 2477.20 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-08-25 12:00:00 | 2495.60 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-08-26 11:30:00 | 2455.00 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-26 14:15:00 | 2459.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-26 15:00:00 | 2458.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-28 09:15:00 | 2435.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-08-29 09:15:00 | 2436.10 | 2025-09-03 09:15:00 | 2503.40 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-09-01 10:00:00 | 2436.20 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-01 11:00:00 | 2440.60 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-09-01 14:30:00 | 2441.50 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-09-02 13:45:00 | 2475.30 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest1 | 2026-02-05 09:15:00 | 2425.60 | 2026-02-10 12:15:00 | 2492.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest1 | 2026-02-05 13:15:00 | 2434.90 | 2026-02-10 12:15:00 | 2492.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest1 | 2026-02-05 15:15:00 | 2420.20 | 2026-02-10 12:15:00 | 2492.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2376.70 | 2026-03-02 09:15:00 | 2257.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2376.70 | 2026-03-05 11:15:00 | 2139.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2378.70 | 2026-04-28 09:15:00 | 2276.49 | PARTIAL | 0.50 | 4.30% |
| SELL | retest2 | 2026-04-15 11:00:00 | 2396.30 | 2026-04-28 09:15:00 | 2282.85 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2378.70 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest2 | 2026-04-15 11:00:00 | 2396.30 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2026-04-17 10:15:00 | 2403.00 | 2026-05-06 09:15:00 | 2350.20 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2313.00 | 2026-05-06 09:15:00 | 2350.20 | STOP_HIT | 1.00 | -1.61% |
