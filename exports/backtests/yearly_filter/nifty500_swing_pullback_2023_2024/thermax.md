# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4693.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 5
- **Avg / median % per leg:** 6.01% / 6.46%
- **Sum % (uncompounded):** 66.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 9 | 81.8% | 4 | 2 | 5 | 6.01% | 66.1% |
| BUY @ 2nd Alert (retest1) | 11 | 9 | 81.8% | 4 | 2 | 5 | 6.01% | 66.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 9 | 81.8% | 4 | 2 | 5 | 6.01% | 66.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 2398.60 | 2248.47 | 2301.10 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=54.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 00:00:00 | 2506.72 | 2260.94 | 2378.69 | T1 booked 50% @ 2506.72 |
| Target hit | 2023-09-13 00:00:00 | 2771.20 | 2388.13 | 2778.01 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 00:00:00 | 2976.40 | 2414.33 | 2810.61 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=104.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 00:00:00 | 3185.46 | 2451.61 | 2925.72 | T1 booked 50% @ 3185.46 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 2976.40 | 2475.09 | 2964.11 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 3128.15 | 2504.01 | 2970.84 | Stage2 pullback-breakout RSI=63 vol=4.4x ATR=118.53 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 2950.35 | 2526.53 | 3003.97 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 00:00:00 | 2998.95 | 2603.79 | 2748.64 | Stage2 pullback-breakout RSI=69 vol=2.1x ATR=96.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 00:00:00 | 3192.54 | 2652.18 | 2969.98 | T1 booked 50% @ 3192.54 |
| Target hit | 2024-01-17 00:00:00 | 3088.60 | 2702.50 | 3099.20 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 3181.25 | 2735.03 | 3092.46 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=96.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 00:00:00 | 3373.84 | 2761.09 | 3136.61 | T1 booked 50% @ 3373.84 |
| Target hit | 2024-03-15 00:00:00 | 3484.05 | 2946.84 | 3566.85 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 3859.85 | 2976.69 | 3615.82 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=159.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 00:00:00 | 4178.39 | 3008.81 | 3734.16 | T1 booked 50% @ 4178.39 |
| Target hit | 2024-04-25 00:00:00 | 4300.30 | 3257.40 | 4405.22 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-17 00:00:00 | 2398.60 | 2023-07-25 00:00:00 | 2506.72 | PARTIAL | 0.50 | 4.51% |
| BUY | retest1 | 2023-07-17 00:00:00 | 2398.60 | 2023-09-13 00:00:00 | 2771.20 | TARGET_HIT | 0.50 | 15.53% |
| BUY | retest1 | 2023-09-22 00:00:00 | 2976.40 | 2023-10-03 00:00:00 | 3185.46 | PARTIAL | 0.50 | 7.02% |
| BUY | retest1 | 2023-09-22 00:00:00 | 2976.40 | 2023-10-09 00:00:00 | 2976.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-17 00:00:00 | 3128.15 | 2023-10-23 00:00:00 | 2950.35 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest1 | 2023-12-18 00:00:00 | 2998.95 | 2024-01-03 00:00:00 | 3192.54 | PARTIAL | 0.50 | 6.46% |
| BUY | retest1 | 2023-12-18 00:00:00 | 2998.95 | 2024-01-17 00:00:00 | 3088.60 | TARGET_HIT | 0.50 | 2.99% |
| BUY | retest1 | 2024-01-31 00:00:00 | 3181.25 | 2024-02-08 00:00:00 | 3373.84 | PARTIAL | 0.50 | 6.05% |
| BUY | retest1 | 2024-01-31 00:00:00 | 3181.25 | 2024-03-15 00:00:00 | 3484.05 | TARGET_HIT | 0.50 | 9.52% |
| BUY | retest1 | 2024-03-21 00:00:00 | 3859.85 | 2024-03-27 00:00:00 | 4178.39 | PARTIAL | 0.50 | 8.25% |
| BUY | retest1 | 2024-03-21 00:00:00 | 3859.85 | 2024-04-25 00:00:00 | 4300.30 | TARGET_HIT | 0.50 | 11.41% |
