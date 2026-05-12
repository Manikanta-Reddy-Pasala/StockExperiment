# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2480.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 14
- **Target hits / Stop hits / Partials:** 0 / 18 / 4
- **Avg / median % per leg:** 0.25% / -1.27%
- **Sum % (uncompounded):** 5.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 8 | 36.4% | 0 | 18 | 4 | 0.25% | 5.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 8 | 36.4% | 0 | 18 | 4 | 0.25% | 5.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 8 | 36.4% | 0 | 18 | 4 | 0.25% | 5.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2671.30 | 3098.00 | 3099.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 2643.40 | 3072.66 | 3086.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 2859.20 | 2845.44 | 2920.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 2859.20 | 2845.44 | 2920.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2628.00 | 2557.23 | 2623.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 2628.00 | 2557.23 | 2623.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2621.00 | 2557.86 | 2623.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 2596.00 | 2557.86 | 2623.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2591.00 | 2558.19 | 2623.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 2580.10 | 2558.19 | 2623.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 2576.00 | 2558.37 | 2623.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 2576.00 | 2558.55 | 2622.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 2580.40 | 2560.84 | 2621.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2451.09 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2447.20 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2447.20 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2451.38 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 2520.00 | 2509.26 | 2565.78 | SL hit (close>ema200) qty=0.50 sl=2509.26 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 2604.50 | 2506.42 | 2506.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 2623.80 | 2509.34 | 2507.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 2399.80 | 2514.88 | 2515.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2390.10 | 2496.18 | 2505.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 2414.00 | 2400.01 | 2443.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 09:45:00 | 2422.60 | 2400.01 | 2443.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 2446.40 | 2400.37 | 2441.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 2446.40 | 2400.37 | 2441.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 2422.00 | 2400.59 | 2441.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 2416.60 | 2401.16 | 2441.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 2420.80 | 2401.45 | 2441.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2408.50 | 2401.86 | 2439.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 2420.80 | 2402.32 | 2439.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2455.60 | 2403.03 | 2439.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 2455.60 | 2403.03 | 2439.15 | SL hit (close>static) qty=1.00 sl=2453.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-12 10:15:00 | 2580.10 | 2025-12-29 10:15:00 | 2451.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:45:00 | 2576.00 | 2025-12-29 10:15:00 | 2447.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 2576.00 | 2025-12-29 10:15:00 | 2447.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 11:45:00 | 2580.40 | 2025-12-29 10:15:00 | 2451.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 2580.10 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2025-12-12 10:45:00 | 2576.00 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-12-12 11:45:00 | 2576.00 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-12-15 11:45:00 | 2580.40 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2026-02-03 12:45:00 | 2413.90 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 2403.50 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-05 15:15:00 | 2425.10 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-13 14:30:00 | 2416.60 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-15 10:15:00 | 2420.80 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-16 09:30:00 | 2408.50 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-16 15:15:00 | 2420.80 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-20 09:15:00 | 2436.70 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-04-20 12:00:00 | 2429.60 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-21 09:45:00 | 2436.90 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-21 10:30:00 | 2436.80 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-04-28 12:00:00 | 2432.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-05-06 12:15:00 | 2431.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-05-06 13:15:00 | 2421.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -1.27% |
