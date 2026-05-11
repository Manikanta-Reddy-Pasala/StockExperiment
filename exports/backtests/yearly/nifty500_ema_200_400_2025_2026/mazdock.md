# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2656.80
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
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 2
- **Target hits / Stop hits / Partials:** 4 / 7 / 9
- **Avg / median % per leg:** 4.57% / 4.99%
- **Sum % (uncompounded):** 91.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 18 | 90.0% | 4 | 7 | 9 | 4.57% | 91.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 18 | 90.0% | 4 | 7 | 9 | 4.57% | 91.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 18 | 90.0% | 4 | 7 | 9 | 4.57% | 91.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2662.90 | 3092.41 | 3093.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2639.00 | 2942.30 | 3007.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 2758.30 | 2757.26 | 2850.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 2758.30 | 2757.26 | 2850.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2886.10 | 2760.73 | 2847.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2886.10 | 2760.73 | 2847.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2898.20 | 2762.10 | 2848.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2879.40 | 2860.81 | 2880.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 2863.90 | 2860.84 | 2880.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2735.43 | 2857.07 | 2877.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:15:00 | 2720.70 | 2848.55 | 2872.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2848.00 | 2846.75 | 2870.75 | SL hit (close>ema200) qty=0.50 sl=2846.75 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 2670.70 | 2408.86 | 2407.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 2688.00 | 2426.39 | 2416.55 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-26 09:15:00 | 2879.40 | 2025-09-29 12:15:00 | 2735.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 10:00:00 | 2863.90 | 2025-09-30 14:15:00 | 2720.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2879.40 | 2025-10-01 15:15:00 | 2848.00 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-09-26 10:00:00 | 2863.90 | 2025-10-01 15:15:00 | 2848.00 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2025-10-06 09:30:00 | 2882.00 | 2025-10-09 15:15:00 | 2898.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-06 14:15:00 | 2885.60 | 2025-10-09 15:15:00 | 2898.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-07 15:00:00 | 2875.90 | 2025-10-29 09:15:00 | 2737.90 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2868.00 | 2025-10-29 09:15:00 | 2741.32 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2875.00 | 2025-10-29 09:15:00 | 2731.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 14:30:00 | 2874.20 | 2025-10-29 09:15:00 | 2730.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 2839.50 | 2025-11-03 09:15:00 | 2711.87 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2025-10-16 10:00:00 | 2854.60 | 2025-11-03 09:15:00 | 2712.25 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-10-28 09:45:00 | 2855.00 | 2025-11-04 12:15:00 | 2697.53 | PARTIAL | 0.50 | 5.52% |
| SELL | retest2 | 2025-10-07 15:00:00 | 2875.90 | 2025-11-07 09:15:00 | 2593.80 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2868.00 | 2025-11-07 09:15:00 | 2597.04 | TARGET_HIT | 0.50 | 9.45% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2875.00 | 2025-11-07 09:15:00 | 2587.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-10 14:30:00 | 2874.20 | 2025-11-07 09:15:00 | 2586.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 2839.50 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2025-10-16 10:00:00 | 2854.60 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-10-28 09:45:00 | 2855.00 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.63% |
