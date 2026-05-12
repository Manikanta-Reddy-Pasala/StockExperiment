# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1875.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 2
- **Avg / median % per leg:** 3.92% / 5.00%
- **Sum % (uncompounded):** 35.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 1 | 4 | 0 | 1.06% | 5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 1 | 4 | 0 | 1.06% | 5.3% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 5 | 55.6% | 3 | 4 | 2 | 3.92% | 35.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 2654.20 | 2830.22 | 2830.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2632.05 | 2797.76 | 2813.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 2774.60 | 2771.48 | 2797.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 12:00:00 | 2774.60 | 2771.48 | 2797.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2801.40 | 2771.78 | 2797.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2801.40 | 2771.78 | 2797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2822.55 | 2772.28 | 2798.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 2801.00 | 2776.11 | 2799.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 2798.50 | 2777.18 | 2799.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 2660.95 | 2771.51 | 2795.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 2658.57 | 2771.51 | 2795.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 12:15:00 | 2520.90 | 2742.24 | 2778.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 1682.80 | 1536.78 | 1536.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1684.00 | 1539.68 | 1537.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-24 10:45:00 | 2361.30 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-05-24 11:45:00 | 2360.00 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-05-27 10:00:00 | 2356.90 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-27 14:00:00 | 2356.55 | 2024-05-28 09:15:00 | 2310.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-06-11 09:15:00 | 2308.00 | 2024-06-20 09:15:00 | 2538.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 2801.00 | 2024-11-11 09:15:00 | 2660.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 2798.50 | 2024-11-11 09:15:00 | 2658.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 2801.00 | 2024-11-13 12:15:00 | 2520.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 2798.50 | 2024-11-13 12:15:00 | 2518.65 | TARGET_HIT | 0.50 | 10.00% |
