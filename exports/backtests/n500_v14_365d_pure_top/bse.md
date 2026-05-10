# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 3905.00
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
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 0
- **Avg / median % per leg:** 3.41% / -1.10%
- **Sum % (uncompounded):** 37.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 5 | 6 | 0 | 3.41% | 37.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 5 | 6 | 0 | 3.41% | 37.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 5 | 45.5% | 5 | 6 | 0 | 3.41% | 37.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 2306.30 | 2465.00 | 2465.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 2290.00 | 2460.08 | 2462.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2366.50 | 2354.16 | 2400.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 2351.20 | 2354.16 | 2400.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2277.00 | 2201.37 | 2274.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2277.00 | 2201.37 | 2274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2318.00 | 2202.53 | 2274.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 2318.00 | 2202.53 | 2274.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2325.00 | 2203.75 | 2274.93 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 2486.50 | 2325.56 | 2325.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2500.00 | 2327.29 | 2326.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 2686.50 | 2734.65 | 2611.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-09 09:45:00 | 2669.70 | 2734.65 | 2611.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2579.90 | 2729.09 | 2615.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 2579.90 | 2729.09 | 2615.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 2582.10 | 2727.63 | 2615.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 2648.10 | 2727.63 | 2615.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 2602.00 | 2717.67 | 2622.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 2595.00 | 2695.00 | 2636.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-13 09:15:00 | 2862.20 | 2696.68 | 2652.58 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-13 09:15:00 | 2854.50 | 2696.68 | 2652.58 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 2550.00 | 2733.88 | 2690.38 | SL hit (close<static) qty=1.00 sl=2575.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 2612.80 | 2732.50 | 2689.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 2573.40 | 2730.91 | 2689.32 | SL hit (close<static) qty=1.00 sl=2575.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2631.00 | 2726.16 | 2687.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 2631.00 | 2726.16 | 2687.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 2664.50 | 2724.16 | 2687.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 2689.80 | 2723.89 | 2687.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 09:15:00 | 2958.78 | 2761.84 | 2712.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 2681.00 | 2784.31 | 2759.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 09:30:00 | 2706.20 | 2780.08 | 2759.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-17 12:15:00 | 2949.10 | 2802.15 | 2775.23 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-17 13:15:00 | 2976.82 | 2803.86 | 2776.22 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-08 15:15:00 | 2491.00 | 2025-07-11 09:15:00 | 2419.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-07-15 11:45:00 | 2481.80 | 2025-07-18 09:15:00 | 2454.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-17 12:15:00 | 2481.30 | 2025-07-18 09:15:00 | 2454.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-21 09:30:00 | 2502.20 | 2025-07-25 14:15:00 | 2452.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-12-11 09:15:00 | 2648.10 | 2026-01-13 09:15:00 | 2862.20 | TARGET_HIT | 1.00 | 8.09% |
| BUY | retest2 | 2025-12-16 10:30:00 | 2602.00 | 2026-01-13 09:15:00 | 2854.50 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2025-12-30 15:15:00 | 2595.00 | 2026-02-01 11:15:00 | 2550.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-02-01 12:45:00 | 2612.80 | 2026-02-01 13:15:00 | 2573.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-02-02 14:45:00 | 2689.80 | 2026-02-09 09:15:00 | 2958.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-05 09:30:00 | 2681.00 | 2026-03-17 12:15:00 | 2949.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 09:30:00 | 2706.20 | 2026-03-17 13:15:00 | 2976.82 | TARGET_HIT | 1.00 | 10.00% |
