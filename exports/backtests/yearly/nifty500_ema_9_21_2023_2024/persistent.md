# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5115.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 219 |
| ALERT1 | 154 |
| ALERT2 | 151 |
| ALERT2_SKIP | 98 |
| ALERT3 | 317 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 126 |
| PARTIAL | 17 |
| TARGET_HIT | 12 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 51 / 92
- **Target hits / Stop hits / Partials:** 12 / 114 / 17
- **Avg / median % per leg:** 0.82% / -0.73%
- **Sum % (uncompounded):** 117.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 19 | 29.2% | 7 | 58 | 0 | 0.47% | 30.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 65 | 19 | 29.2% | 7 | 58 | 0 | 0.47% | 30.4% |
| SELL (all) | 78 | 32 | 41.0% | 5 | 56 | 17 | 1.12% | 87.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.40% | 10.8% |
| SELL @ 3rd Alert (retest2) | 76 | 30 | 39.5% | 5 | 55 | 16 | 1.01% | 76.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.40% | 10.8% |
| retest2 (combined) | 141 | 49 | 34.8% | 12 | 113 | 16 | 0.76% | 106.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 4692.65 | 4676.38 | 4675.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 15:15:00 | 4699.00 | 4682.15 | 4678.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 14:15:00 | 4717.40 | 4729.38 | 4709.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 10:15:00 | 4726.00 | 4731.88 | 4715.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 4726.00 | 4731.88 | 4715.31 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 4983.55 | 5168.22 | 5175.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 4923.90 | 5119.36 | 5153.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 13:15:00 | 5012.85 | 5007.99 | 5052.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 4969.95 | 5002.27 | 5039.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 4969.95 | 5002.27 | 5039.14 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 15:15:00 | 4977.00 | 4952.78 | 4949.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 5058.00 | 4973.82 | 4959.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 4900.70 | 5001.98 | 4989.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 4900.70 | 5001.98 | 4989.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 4900.70 | 5001.98 | 4989.06 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 11:15:00 | 4928.25 | 4971.88 | 4976.67 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 11:15:00 | 5003.85 | 4979.29 | 4976.00 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 09:15:00 | 4954.00 | 4975.95 | 4976.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 09:15:00 | 4930.00 | 4953.15 | 4962.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-19 11:15:00 | 4968.40 | 4955.23 | 4961.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 11:15:00 | 4968.40 | 4955.23 | 4961.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 4968.40 | 4955.23 | 4961.65 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 15:15:00 | 4971.00 | 4954.33 | 4954.11 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 4938.60 | 4952.70 | 4953.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 4904.95 | 4938.28 | 4946.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 4834.85 | 4826.80 | 4856.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 4912.00 | 4846.57 | 4858.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 4912.00 | 4846.57 | 4858.56 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 4906.60 | 4868.73 | 4867.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 5029.00 | 4912.75 | 4895.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 4902.60 | 4955.83 | 4931.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 4902.60 | 4955.83 | 4931.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 4902.60 | 4955.83 | 4931.75 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 12:15:00 | 4877.00 | 4914.71 | 4916.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 13:15:00 | 4872.20 | 4906.21 | 4912.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 13:15:00 | 4861.15 | 4858.97 | 4880.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 15:15:00 | 4877.70 | 4863.85 | 4879.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 4877.70 | 4863.85 | 4879.32 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 14:15:00 | 4887.00 | 4864.39 | 4864.28 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 4831.70 | 4862.71 | 4864.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 4783.60 | 4832.91 | 4848.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 13:15:00 | 4702.50 | 4696.68 | 4733.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 4854.30 | 4727.37 | 4738.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 4854.30 | 4727.37 | 4738.35 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 4867.95 | 4755.49 | 4750.13 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 09:15:00 | 2400.32 | 4294.59 | 4546.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 13:15:00 | 2375.57 | 2438.68 | 2523.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 10:15:00 | 2355.20 | 2350.50 | 2387.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 13:15:00 | 2356.75 | 2342.12 | 2357.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 2356.75 | 2342.12 | 2357.88 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 2371.85 | 2361.49 | 2360.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 2382.50 | 2369.33 | 2364.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 2376.50 | 2388.68 | 2378.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 2376.50 | 2388.68 | 2378.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 2376.50 | 2388.68 | 2378.83 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 2336.82 | 2369.53 | 2372.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 11:15:00 | 2328.93 | 2349.90 | 2360.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 2346.50 | 2341.64 | 2353.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 15:15:00 | 2351.13 | 2343.54 | 2353.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 2351.13 | 2343.54 | 2353.38 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 2378.50 | 2361.55 | 2360.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 14:15:00 | 2397.00 | 2373.35 | 2366.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 12:15:00 | 2431.00 | 2431.32 | 2410.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 2407.75 | 2426.22 | 2414.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 2407.75 | 2426.22 | 2414.70 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 2418.05 | 2459.17 | 2463.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 10:15:00 | 2389.00 | 2445.13 | 2456.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 15:15:00 | 2432.50 | 2428.44 | 2442.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 2457.00 | 2434.15 | 2443.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 2457.00 | 2434.15 | 2443.60 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 2484.00 | 2452.66 | 2450.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 2502.78 | 2462.68 | 2455.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 15:15:00 | 2509.98 | 2511.26 | 2493.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 12:15:00 | 2501.00 | 2515.99 | 2501.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 2501.00 | 2515.99 | 2501.87 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 09:15:00 | 1283.78 | 2300.85 | 2425.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 12:15:00 | 1273.74 | 1803.12 | 2141.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 1335.58 | 1309.11 | 1444.15 | EMA200 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 2712.20 | 1607.21 | 1517.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 10:15:00 | 2736.13 | 1832.99 | 1627.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 13:15:00 | 2967.58 | 2968.36 | 2882.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 2954.23 | 2973.41 | 2957.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 2954.23 | 2973.41 | 2957.41 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 2920.35 | 2948.52 | 2950.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 2872.75 | 2928.40 | 2940.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 2942.10 | 2902.53 | 2916.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 2942.10 | 2902.53 | 2916.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 2942.10 | 2902.53 | 2916.07 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 2966.50 | 2930.69 | 2926.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 2985.48 | 2963.40 | 2949.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 2959.45 | 2963.60 | 2952.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 2959.45 | 2963.60 | 2952.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 2959.45 | 2963.60 | 2952.25 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 2918.63 | 2942.09 | 2945.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 2883.23 | 2926.41 | 2937.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 11:15:00 | 2919.00 | 2903.01 | 2914.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 11:15:00 | 2919.00 | 2903.01 | 2914.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 2919.00 | 2903.01 | 2914.74 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 2922.05 | 2907.64 | 2906.86 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 09:15:00 | 2891.00 | 2904.31 | 2905.42 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 2941.75 | 2911.30 | 2908.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 2961.83 | 2921.41 | 2913.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 10:15:00 | 2927.58 | 2935.47 | 2924.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 10:15:00 | 2927.58 | 2935.47 | 2924.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 2927.58 | 2935.47 | 2924.94 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 14:15:00 | 2919.73 | 2926.09 | 2926.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 2905.23 | 2921.10 | 2923.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 2877.90 | 2864.68 | 2886.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 12:15:00 | 2907.48 | 2873.24 | 2888.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 2907.48 | 2873.24 | 2888.01 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 14:15:00 | 2863.13 | 2696.77 | 2683.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 2883.20 | 2859.48 | 2822.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 13:15:00 | 2867.55 | 2869.76 | 2837.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 2839.98 | 2862.67 | 2844.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 2839.98 | 2862.67 | 2844.54 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 2793.40 | 2831.03 | 2833.87 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 2890.35 | 2843.64 | 2838.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 2924.03 | 2876.34 | 2857.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 2886.70 | 2890.98 | 2872.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 14:15:00 | 2881.18 | 2889.02 | 2872.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 2881.18 | 2889.02 | 2872.97 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 2847.60 | 2874.61 | 2877.11 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 11:15:00 | 2880.88 | 2876.84 | 2876.57 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 12:15:00 | 2869.33 | 2875.34 | 2875.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 14:15:00 | 2862.93 | 2872.51 | 2874.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 15:15:00 | 2887.13 | 2875.43 | 2875.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 15:15:00 | 2887.13 | 2875.43 | 2875.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 2887.13 | 2875.43 | 2875.65 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 09:15:00 | 2963.50 | 2867.16 | 2862.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 10:15:00 | 2968.08 | 2927.09 | 2903.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 09:15:00 | 2940.85 | 2941.37 | 2922.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 11:15:00 | 2921.68 | 2938.42 | 2924.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 2921.68 | 2938.42 | 2924.26 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 2864.10 | 3050.87 | 3064.83 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 3042.98 | 3010.99 | 3010.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 3056.00 | 3037.85 | 3028.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 11:15:00 | 3097.05 | 3100.31 | 3083.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 14:15:00 | 3121.28 | 3138.86 | 3127.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 3121.28 | 3138.86 | 3127.43 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 10:15:00 | 3082.00 | 3117.18 | 3119.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 11:15:00 | 3072.03 | 3108.15 | 3115.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 14:15:00 | 3121.08 | 3102.73 | 3110.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 14:15:00 | 3121.08 | 3102.73 | 3110.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 3121.08 | 3102.73 | 3110.26 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 3124.70 | 3115.26 | 3114.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 14:15:00 | 3146.88 | 3122.34 | 3117.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 09:15:00 | 3135.03 | 3147.00 | 3137.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 3135.03 | 3147.00 | 3137.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 3135.03 | 3147.00 | 3137.21 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 15:15:00 | 3125.00 | 3133.95 | 3134.22 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 3138.85 | 3134.93 | 3134.65 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 09:15:00 | 3089.70 | 3125.89 | 3130.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 10:15:00 | 3076.38 | 3115.98 | 3125.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 14:15:00 | 3118.08 | 3102.61 | 3114.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 14:15:00 | 3118.08 | 3102.61 | 3114.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 3118.08 | 3102.61 | 3114.76 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 3134.85 | 3122.40 | 3121.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 3153.25 | 3130.36 | 3125.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 13:15:00 | 3184.98 | 3188.56 | 3168.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 15:15:00 | 3220.00 | 3239.41 | 3223.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 3220.00 | 3239.41 | 3223.65 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 14:15:00 | 3187.83 | 3243.98 | 3246.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 3184.13 | 3212.90 | 3228.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 15:15:00 | 3167.83 | 3166.90 | 3189.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 3218.03 | 3177.13 | 3191.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 3218.03 | 3177.13 | 3191.75 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 3217.65 | 3200.23 | 3200.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 3243.50 | 3216.46 | 3208.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 10:15:00 | 3213.50 | 3215.87 | 3208.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 11:15:00 | 3235.48 | 3219.79 | 3211.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 3235.48 | 3219.79 | 3211.15 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 14:15:00 | 3174.50 | 3207.17 | 3207.30 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 3213.28 | 3206.62 | 3206.47 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 15:15:00 | 3200.53 | 3205.40 | 3205.93 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 3230.98 | 3210.52 | 3208.20 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 3147.80 | 3198.28 | 3204.83 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 13:15:00 | 3247.50 | 3186.42 | 3181.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 12:15:00 | 3252.78 | 3226.96 | 3207.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 14:15:00 | 3263.33 | 3273.86 | 3249.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 3244.28 | 3266.13 | 3250.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 3244.28 | 3266.13 | 3250.09 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 09:15:00 | 3684.10 | 3698.82 | 3698.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 10:15:00 | 3666.13 | 3692.28 | 3695.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 3666.23 | 3661.16 | 3673.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 14:15:00 | 3640.90 | 3628.63 | 3646.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 3640.90 | 3628.63 | 3646.05 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 3670.70 | 3648.85 | 3647.20 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 3613.98 | 3648.14 | 3649.04 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 3677.50 | 3650.16 | 3648.67 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 3634.00 | 3648.76 | 3650.32 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 3700.00 | 3656.12 | 3653.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 3724.60 | 3675.87 | 3662.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 15:15:00 | 3692.00 | 3697.59 | 3682.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 3802.50 | 3828.90 | 3798.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 3802.50 | 3828.90 | 3798.53 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 10:15:00 | 4189.00 | 4222.39 | 4222.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 13:15:00 | 4143.00 | 4191.86 | 4207.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 4226.33 | 4161.91 | 4174.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 4226.33 | 4161.91 | 4174.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 4226.33 | 4161.91 | 4174.49 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 4251.50 | 4188.96 | 4185.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 4278.20 | 4232.14 | 4209.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 11:15:00 | 4276.88 | 4283.59 | 4254.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 4285.55 | 4309.48 | 4297.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 4285.55 | 4309.48 | 4297.39 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 4275.75 | 4314.32 | 4315.93 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 4352.90 | 4322.75 | 4319.33 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 4294.00 | 4317.17 | 4317.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 11:15:00 | 4272.50 | 4308.23 | 4313.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 15:15:00 | 4329.00 | 4309.55 | 4312.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 15:15:00 | 4329.00 | 4309.55 | 4312.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 4329.00 | 4309.55 | 4312.12 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 4362.48 | 4320.14 | 4316.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 14:15:00 | 4435.35 | 4374.85 | 4354.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 12:15:00 | 4381.98 | 4395.73 | 4375.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 13:15:00 | 4380.20 | 4392.62 | 4375.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 4380.20 | 4392.62 | 4375.56 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 4328.27 | 4368.31 | 4368.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 09:15:00 | 4321.02 | 4347.94 | 4357.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 4280.45 | 4276.72 | 4299.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 4335.98 | 4289.90 | 4301.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 4335.98 | 4289.90 | 4301.74 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 4349.45 | 4316.57 | 4312.32 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 4198.10 | 4294.18 | 4303.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 4155.35 | 4266.42 | 4290.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 4226.40 | 4221.76 | 4251.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 12:15:00 | 4258.50 | 4232.27 | 4249.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 4258.50 | 4232.27 | 4249.42 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 4255.00 | 4238.45 | 4237.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 4330.25 | 4256.81 | 4245.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 09:15:00 | 4291.50 | 4295.36 | 4279.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 4291.50 | 4295.36 | 4279.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 4291.50 | 4295.36 | 4279.02 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 4228.77 | 4267.63 | 4271.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 4141.10 | 4224.18 | 4247.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 4096.02 | 4092.61 | 4144.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 4186.18 | 4111.32 | 4147.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 4186.18 | 4111.32 | 4147.95 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 13:15:00 | 4166.75 | 4163.92 | 4163.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 14:15:00 | 4186.27 | 4168.39 | 4165.88 | Break + close above crossover candle high |

### Cycle 70 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 4130.00 | 4160.73 | 4162.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 4119.15 | 4145.33 | 4153.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 14:15:00 | 4098.55 | 4094.27 | 4118.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 14:15:00 | 4098.55 | 4094.27 | 4118.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 4098.55 | 4094.27 | 4118.49 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 4162.80 | 4103.56 | 4102.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 15:15:00 | 4171.95 | 4117.24 | 4109.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 4104.45 | 4114.68 | 4108.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 4104.45 | 4114.68 | 4108.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 4104.45 | 4114.68 | 4108.81 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 11:15:00 | 4069.70 | 4126.27 | 4126.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 4006.45 | 4083.55 | 4104.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 4010.00 | 4006.35 | 4044.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 4092.50 | 4025.19 | 4037.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 4092.50 | 4025.19 | 4037.19 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 4070.00 | 4045.71 | 4044.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 4104.08 | 4059.44 | 4051.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 09:15:00 | 4001.45 | 4054.37 | 4050.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 09:15:00 | 4001.45 | 4054.37 | 4050.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 4001.45 | 4054.37 | 4050.75 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 10:15:00 | 4012.50 | 4046.00 | 4047.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 12:15:00 | 3989.25 | 4027.77 | 4038.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 4029.20 | 4012.81 | 4026.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 4029.20 | 4012.81 | 4026.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 4029.20 | 4012.81 | 4026.09 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 13:15:00 | 4061.65 | 4032.34 | 4031.78 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 09:15:00 | 3983.75 | 4031.38 | 4033.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-01 09:15:00 | 3972.85 | 4006.70 | 4020.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 13:15:00 | 3988.05 | 3986.62 | 4004.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 13:15:00 | 3988.05 | 3986.62 | 4004.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 3988.05 | 3986.62 | 4004.36 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 10:15:00 | 4023.55 | 3982.83 | 3982.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 12:15:00 | 4030.80 | 3998.75 | 3990.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 3995.80 | 4012.95 | 4001.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 3995.80 | 4012.95 | 4001.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 3995.80 | 4012.95 | 4001.42 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 3982.20 | 3995.20 | 3995.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 10:15:00 | 3965.60 | 3984.88 | 3990.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 3950.90 | 3939.46 | 3962.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 3950.90 | 3939.46 | 3962.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 3950.90 | 3939.46 | 3962.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 4011.05 | 3956.23 | 3958.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 4022.95 | 3969.57 | 3964.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 11:15:00 | 4036.45 | 3991.34 | 3975.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 3972.70 | 3992.37 | 3980.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 14:15:00 | 3972.70 | 3992.37 | 3980.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 3972.70 | 3992.37 | 3980.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 3972.70 | 3992.37 | 3980.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 3984.05 | 3990.70 | 3980.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 3956.60 | 3990.70 | 3980.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 3950.10 | 3982.58 | 3978.08 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 3927.50 | 3971.57 | 3973.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 11:15:00 | 3913.50 | 3959.95 | 3968.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 14:15:00 | 3954.90 | 3947.87 | 3959.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 14:15:00 | 3954.90 | 3947.87 | 3959.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 3954.90 | 3947.87 | 3959.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 3954.90 | 3947.87 | 3959.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 3951.20 | 3947.28 | 3957.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 3969.00 | 3947.28 | 3957.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 3932.30 | 3944.28 | 3954.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 3940.10 | 3944.28 | 3954.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 3901.30 | 3899.24 | 3924.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:45:00 | 3836.55 | 3887.82 | 3910.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 3806.60 | 3886.25 | 3907.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 10:45:00 | 3835.95 | 3871.70 | 3896.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 3757.55 | 3881.40 | 3891.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 3682.85 | 3841.69 | 3872.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 10:15:00 | 3624.65 | 3841.69 | 3872.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:15:00 | 3644.72 | 3786.96 | 3845.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:15:00 | 3616.27 | 3786.96 | 3845.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:15:00 | 3644.15 | 3786.96 | 3845.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:15:00 | 3569.67 | 3786.96 | 3845.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-04-22 14:15:00 | 3452.90 | 3632.51 | 3744.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 12:15:00 | 3424.75 | 3401.50 | 3398.94 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 3382.25 | 3395.41 | 3396.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 3359.20 | 3384.47 | 3391.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 12:15:00 | 3367.00 | 3358.65 | 3370.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 12:45:00 | 3365.20 | 3358.65 | 3370.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 3379.70 | 3362.86 | 3371.51 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 10:15:00 | 3408.35 | 3376.77 | 3375.92 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 3352.00 | 3373.21 | 3375.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 3317.45 | 3362.06 | 3370.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 14:15:00 | 3359.80 | 3332.87 | 3349.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 14:15:00 | 3359.80 | 3332.87 | 3349.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 3359.80 | 3332.87 | 3349.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 3359.80 | 3332.87 | 3349.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 3369.05 | 3340.11 | 3351.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:30:00 | 3322.00 | 3332.74 | 3346.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 3392.00 | 3332.22 | 3338.65 | SL hit (close>static) qty=1.00 sl=3370.50 alert=retest2 |

### Cycle 85 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 3395.80 | 3344.93 | 3343.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 3419.00 | 3384.55 | 3365.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 3528.40 | 3530.37 | 3506.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 11:45:00 | 3530.00 | 3530.37 | 3506.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 3492.95 | 3522.51 | 3507.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 3487.55 | 3522.51 | 3507.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 3490.00 | 3516.00 | 3505.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 3478.00 | 3516.00 | 3505.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 3507.45 | 3505.63 | 3502.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 3523.35 | 3508.68 | 3504.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 3530.00 | 3511.14 | 3505.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:45:00 | 3524.00 | 3513.38 | 3507.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 14:15:00 | 3498.25 | 3511.26 | 3508.48 | SL hit (close<static) qty=1.00 sl=3500.15 alert=retest2 |

### Cycle 86 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 3581.15 | 3643.70 | 3650.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 3566.10 | 3619.59 | 3638.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 15:15:00 | 3439.00 | 3432.73 | 3474.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 3334.50 | 3432.73 | 3474.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 3451.50 | 3381.26 | 3414.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 3451.50 | 3381.26 | 3414.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 3573.10 | 3419.63 | 3428.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 3553.75 | 3419.63 | 3428.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 3570.75 | 3449.85 | 3441.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 3586.55 | 3496.50 | 3465.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 3709.60 | 3786.14 | 3708.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 3709.60 | 3786.14 | 3708.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 3709.60 | 3786.14 | 3708.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 3709.60 | 3786.14 | 3708.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 3692.00 | 3767.31 | 3706.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 3696.30 | 3767.31 | 3706.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 3698.30 | 3753.51 | 3706.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:30:00 | 3692.85 | 3753.51 | 3706.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 3703.65 | 3743.54 | 3705.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 15:00:00 | 3750.00 | 3739.84 | 3710.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-28 11:15:00 | 4125.00 | 4056.08 | 4015.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 4583.90 | 4656.89 | 4662.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 12:15:00 | 4553.00 | 4601.20 | 4624.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 13:15:00 | 4599.95 | 4565.14 | 4587.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 13:15:00 | 4599.95 | 4565.14 | 4587.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 4599.95 | 4565.14 | 4587.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:45:00 | 4607.00 | 4565.14 | 4587.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 4620.85 | 4576.28 | 4590.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 4630.05 | 4576.28 | 4590.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 4679.00 | 4603.01 | 4600.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 4776.30 | 4637.67 | 4616.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 13:15:00 | 4780.00 | 4790.00 | 4738.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:00:00 | 4780.00 | 4790.00 | 4738.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 4767.75 | 4806.76 | 4784.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 4764.45 | 4806.76 | 4784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 4809.10 | 4807.23 | 4786.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:30:00 | 4836.90 | 4808.89 | 4789.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:30:00 | 4851.35 | 4818.99 | 4795.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 4654.55 | 4817.05 | 4805.84 | SL hit (close<static) qty=1.00 sl=4759.90 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 4613.30 | 4776.30 | 4788.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 4598.10 | 4714.64 | 4756.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 4709.20 | 4660.58 | 4707.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 4709.20 | 4660.58 | 4707.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 4709.20 | 4660.58 | 4707.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 4709.20 | 4660.58 | 4707.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 4715.00 | 4671.46 | 4707.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 4747.95 | 4671.46 | 4707.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 4762.60 | 4689.69 | 4712.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 4748.75 | 4689.69 | 4712.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 4748.55 | 4701.46 | 4716.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 4787.15 | 4701.46 | 4716.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 4762.00 | 4730.64 | 4727.21 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 4662.55 | 4721.10 | 4723.72 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 4814.00 | 4739.68 | 4731.93 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 4723.10 | 4757.03 | 4759.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 15:15:00 | 4701.00 | 4739.67 | 4750.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 4807.90 | 4753.32 | 4756.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 4807.90 | 4753.32 | 4756.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 4807.90 | 4753.32 | 4756.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 4807.90 | 4753.32 | 4756.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 4808.45 | 4764.34 | 4760.81 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 4737.00 | 4766.30 | 4767.37 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 4782.00 | 4768.84 | 4768.30 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 4764.30 | 4767.93 | 4767.94 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 4778.10 | 4769.96 | 4768.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 4912.90 | 4799.26 | 4782.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 4835.50 | 4847.84 | 4816.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 13:45:00 | 4845.75 | 4847.84 | 4816.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 4823.50 | 4838.05 | 4818.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 4813.30 | 4838.05 | 4818.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 4834.65 | 4837.37 | 4820.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 4834.65 | 4837.37 | 4820.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 4789.70 | 4827.83 | 4817.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 4789.70 | 4827.83 | 4817.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 4802.50 | 4822.77 | 4816.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 4789.85 | 4822.77 | 4816.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 4794.95 | 4809.90 | 4811.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 4715.00 | 4790.92 | 4802.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 4557.25 | 4545.66 | 4630.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 15:00:00 | 4557.25 | 4545.66 | 4630.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4594.35 | 4554.81 | 4619.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 4497.00 | 4546.84 | 4592.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 4648.40 | 4584.99 | 4594.27 | SL hit (close>static) qty=1.00 sl=4636.35 alert=retest2 |

### Cycle 101 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 4673.85 | 4602.76 | 4601.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 4689.00 | 4647.21 | 4626.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 4619.55 | 4641.68 | 4626.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 4619.55 | 4641.68 | 4626.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 4619.55 | 4641.68 | 4626.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 4619.55 | 4641.68 | 4626.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 4616.05 | 4636.55 | 4625.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:30:00 | 4616.20 | 4636.55 | 4625.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 4612.95 | 4631.83 | 4624.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:15:00 | 4602.00 | 4631.83 | 4624.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 4645.00 | 4674.67 | 4657.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:15:00 | 4735.00 | 4675.82 | 4662.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 4744.00 | 4674.18 | 4664.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:15:00 | 4714.60 | 4707.37 | 4686.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 15:15:00 | 4710.00 | 4704.86 | 4688.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 4754.30 | 4715.57 | 4696.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 4727.00 | 4715.57 | 4696.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 4912.70 | 4925.37 | 4890.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 4944.30 | 4925.85 | 4894.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:15:00 | 4935.20 | 4925.85 | 4894.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:00:00 | 4938.80 | 4928.44 | 4898.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 4935.85 | 4919.01 | 4902.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4875.50 | 4932.61 | 4922.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 4875.50 | 4932.61 | 4922.48 | SL hit (close<static) qty=1.00 sl=4890.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 4902.00 | 4915.02 | 4915.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 4883.80 | 4904.98 | 4910.93 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 5035.00 | 4927.78 | 4920.06 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 4905.00 | 4959.93 | 4960.17 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 5013.45 | 4963.22 | 4961.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 11:15:00 | 5025.70 | 4975.72 | 4967.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 15:15:00 | 4992.95 | 4994.19 | 4980.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:15:00 | 4989.95 | 4994.19 | 4980.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 5015.90 | 4998.54 | 4983.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 5039.90 | 4998.54 | 4983.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 13:15:00 | 5042.55 | 5014.09 | 4995.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 5186.05 | 5231.94 | 5237.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 5186.05 | 5231.94 | 5237.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 5148.25 | 5210.05 | 5226.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 5189.90 | 5188.92 | 5206.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 5252.00 | 5188.92 | 5206.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 5224.95 | 5196.12 | 5208.08 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 5281.10 | 5223.13 | 5218.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 5297.50 | 5238.00 | 5225.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 5281.55 | 5299.79 | 5270.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 5281.55 | 5299.79 | 5270.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 5292.80 | 5300.72 | 5278.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 5282.25 | 5300.72 | 5278.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 5287.00 | 5295.76 | 5280.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 5302.25 | 5297.06 | 5282.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:30:00 | 5298.50 | 5300.44 | 5285.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 5301.95 | 5299.82 | 5286.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 5158.75 | 5314.64 | 5322.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 5158.75 | 5314.64 | 5322.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 5132.15 | 5278.14 | 5305.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 5209.00 | 5203.99 | 5248.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 5209.00 | 5203.99 | 5248.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 5209.00 | 5203.99 | 5248.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 5314.95 | 5203.99 | 5248.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 5208.30 | 5207.07 | 5238.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:30:00 | 5248.55 | 5207.07 | 5238.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 5210.85 | 5207.83 | 5236.28 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 5345.65 | 5260.41 | 5255.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 5425.10 | 5323.01 | 5287.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 13:15:00 | 5356.45 | 5358.34 | 5329.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:00:00 | 5356.45 | 5358.34 | 5329.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 5330.00 | 5352.67 | 5329.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:30:00 | 5330.70 | 5352.67 | 5329.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 5317.00 | 5345.54 | 5328.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 5351.85 | 5345.54 | 5328.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 5290.55 | 5334.54 | 5325.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 5290.55 | 5334.54 | 5325.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 5332.85 | 5334.20 | 5325.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:30:00 | 5285.60 | 5334.20 | 5325.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 5328.45 | 5333.05 | 5325.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 5323.30 | 5333.05 | 5325.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 5319.80 | 5330.40 | 5325.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:30:00 | 5316.10 | 5330.40 | 5325.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 5377.55 | 5339.83 | 5330.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 14:15:00 | 5385.55 | 5339.83 | 5330.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 5320.25 | 5335.54 | 5335.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 5320.25 | 5335.54 | 5335.82 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 5398.95 | 5346.53 | 5340.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 10:15:00 | 5404.00 | 5358.03 | 5346.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 5435.35 | 5440.68 | 5410.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 15:00:00 | 5435.35 | 5440.68 | 5410.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 5474.90 | 5447.47 | 5418.55 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 5297.70 | 5418.70 | 5429.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 5245.35 | 5384.03 | 5413.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 5335.40 | 5306.27 | 5356.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:00:00 | 5335.40 | 5306.27 | 5356.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 5345.70 | 5314.16 | 5355.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 5345.70 | 5314.16 | 5355.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 5205.00 | 5292.33 | 5341.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:15:00 | 5199.00 | 5292.33 | 5341.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 5197.55 | 5204.52 | 5270.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 5195.45 | 5219.15 | 5248.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 5291.20 | 5264.39 | 5261.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 5291.20 | 5264.39 | 5261.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 5305.60 | 5272.63 | 5265.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 5331.60 | 5335.80 | 5313.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:30:00 | 5332.70 | 5335.80 | 5313.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 5287.50 | 5326.14 | 5311.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 5287.50 | 5326.14 | 5311.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 5219.60 | 5304.83 | 5302.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 5219.60 | 5304.83 | 5302.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 5237.00 | 5291.27 | 5296.79 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 5343.00 | 5305.52 | 5301.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 5393.05 | 5323.02 | 5309.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 5567.40 | 5605.15 | 5553.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 12:00:00 | 5567.40 | 5605.15 | 5553.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 5550.00 | 5594.12 | 5552.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:45:00 | 5550.00 | 5594.12 | 5552.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 5596.00 | 5594.50 | 5556.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:30:00 | 5600.35 | 5584.80 | 5555.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 5490.65 | 5562.00 | 5550.17 | SL hit (close<static) qty=1.00 sl=5545.25 alert=retest2 |

### Cycle 116 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 5418.60 | 5522.89 | 5535.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 5400.25 | 5485.71 | 5504.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 5674.00 | 5323.67 | 5359.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 5674.00 | 5323.67 | 5359.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 5674.00 | 5323.67 | 5359.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 5674.00 | 5323.67 | 5359.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 5738.00 | 5406.54 | 5393.46 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 5602.55 | 5642.43 | 5643.01 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 5667.10 | 5643.52 | 5642.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 5710.00 | 5661.00 | 5651.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 5604.45 | 5655.75 | 5652.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 5604.45 | 5655.75 | 5652.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 5604.45 | 5655.75 | 5652.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 5604.45 | 5655.75 | 5652.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 5621.80 | 5648.96 | 5649.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 5393.50 | 5592.44 | 5623.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 11:15:00 | 5424.30 | 5407.78 | 5465.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 12:00:00 | 5424.30 | 5407.78 | 5465.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 5373.05 | 5408.74 | 5456.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:30:00 | 5398.40 | 5408.74 | 5456.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 5320.35 | 5377.78 | 5429.39 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 5644.25 | 5441.06 | 5433.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 5690.95 | 5522.86 | 5473.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 13:15:00 | 5704.20 | 5719.73 | 5667.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:30:00 | 5700.00 | 5719.73 | 5667.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 5660.40 | 5699.76 | 5667.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 5685.00 | 5699.76 | 5667.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 5692.65 | 5698.34 | 5669.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 5728.55 | 5704.38 | 5674.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 12:00:00 | 5731.00 | 5709.70 | 5680.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 13:30:00 | 5732.05 | 5714.36 | 5687.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:45:00 | 5729.10 | 5719.05 | 5696.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 5737.30 | 5732.28 | 5710.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 5714.65 | 5732.28 | 5710.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 5677.60 | 5721.34 | 5707.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 5677.60 | 5721.34 | 5707.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 5673.00 | 5711.67 | 5704.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 5666.85 | 5711.67 | 5704.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 5613.55 | 5692.05 | 5696.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 5613.55 | 5692.05 | 5696.40 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 5726.30 | 5692.60 | 5688.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 13:15:00 | 5738.10 | 5701.70 | 5692.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 09:15:00 | 5639.60 | 5694.44 | 5692.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 5639.60 | 5694.44 | 5692.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 5639.60 | 5694.44 | 5692.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:45:00 | 5605.85 | 5694.44 | 5692.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 5604.95 | 5676.54 | 5684.34 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 5780.55 | 5678.59 | 5678.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5887.30 | 5798.15 | 5761.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 5944.40 | 5951.00 | 5897.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 5980.00 | 5951.00 | 5897.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 5905.45 | 5948.86 | 5922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:30:00 | 5903.90 | 5948.86 | 5922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 5914.00 | 5941.89 | 5921.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 5860.50 | 5941.89 | 5921.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 5830.45 | 5906.09 | 5908.28 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 5933.00 | 5890.35 | 5889.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 10:15:00 | 5938.00 | 5907.62 | 5899.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 5868.70 | 5917.98 | 5911.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 5868.70 | 5917.98 | 5911.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 5868.70 | 5917.98 | 5911.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 5868.70 | 5917.98 | 5911.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 5918.50 | 5918.09 | 5912.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:30:00 | 5941.85 | 5921.24 | 5914.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:30:00 | 5935.00 | 5920.82 | 5914.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:30:00 | 5939.65 | 5926.66 | 5917.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-17 09:15:00 | 6536.04 | 6492.28 | 6452.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 6434.95 | 6550.06 | 6551.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 6385.40 | 6517.13 | 6536.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 6448.10 | 6447.86 | 6491.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:45:00 | 6446.90 | 6447.86 | 6491.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 6436.85 | 6445.66 | 6486.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 6477.15 | 6445.66 | 6486.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 6399.00 | 6357.08 | 6387.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 6399.00 | 6357.08 | 6387.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 6397.90 | 6365.24 | 6388.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:30:00 | 6407.65 | 6365.24 | 6388.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 6362.00 | 6364.59 | 6386.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 6380.00 | 6364.59 | 6386.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 6391.40 | 6369.96 | 6386.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 6364.50 | 6372.16 | 6386.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 6340.85 | 6384.44 | 6387.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 6415.85 | 6393.76 | 6391.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 6415.85 | 6393.76 | 6391.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 6443.00 | 6403.60 | 6395.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 6280.75 | 6426.92 | 6416.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 6280.75 | 6426.92 | 6416.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 6280.75 | 6426.92 | 6416.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 6280.75 | 6426.92 | 6416.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 6358.15 | 6413.16 | 6410.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:15:00 | 6363.00 | 6413.16 | 6410.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:45:00 | 6405.10 | 6413.96 | 6411.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 6369.15 | 6421.29 | 6421.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 11:15:00 | 6369.15 | 6421.29 | 6421.55 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 6436.75 | 6423.09 | 6421.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 6459.30 | 6431.47 | 6426.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 6444.30 | 6452.42 | 6441.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 6444.30 | 6452.42 | 6441.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 6444.30 | 6452.42 | 6441.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 6444.30 | 6452.42 | 6441.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 6333.15 | 6428.57 | 6431.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 6296.90 | 6376.69 | 6401.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 6346.95 | 6338.51 | 6370.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 6346.95 | 6338.51 | 6370.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 6346.95 | 6338.51 | 6370.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 6378.35 | 6338.51 | 6370.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 6374.55 | 6345.71 | 6371.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 6374.55 | 6345.71 | 6371.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 6355.95 | 6347.76 | 6369.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 13:45:00 | 6340.00 | 6357.56 | 6370.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:00:00 | 6326.35 | 6351.32 | 6366.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 11:15:00 | 6398.00 | 6258.76 | 6255.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 6398.00 | 6258.76 | 6255.22 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 6211.25 | 6266.93 | 6267.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 6096.50 | 6232.85 | 6252.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 6008.00 | 5969.46 | 6055.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:30:00 | 5964.65 | 5969.46 | 6055.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 6096.95 | 5994.96 | 6059.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 6118.00 | 5994.96 | 6059.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 6202.45 | 6036.45 | 6072.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 6202.45 | 6036.45 | 6072.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 6133.00 | 6093.07 | 6090.88 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 14:15:00 | 6059.75 | 6088.69 | 6090.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 5985.50 | 6066.09 | 6080.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 6021.05 | 5976.47 | 6015.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 6021.05 | 5976.47 | 6015.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 6021.05 | 5976.47 | 6015.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 6021.05 | 5976.47 | 6015.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 6034.30 | 5988.04 | 6017.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:15:00 | 6056.95 | 5988.04 | 6017.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 6026.85 | 5995.80 | 6018.12 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 6073.15 | 6030.80 | 6030.14 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 5948.00 | 6018.40 | 6025.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 5944.50 | 5996.59 | 6013.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 5696.45 | 5692.85 | 5816.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 14:45:00 | 5700.00 | 5692.85 | 5816.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 6251.45 | 5798.51 | 5842.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 6251.45 | 5798.51 | 5842.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 6227.25 | 5884.26 | 5877.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 6343.15 | 5976.04 | 5920.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 6168.60 | 6304.95 | 6205.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 6168.60 | 6304.95 | 6205.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 6168.60 | 6304.95 | 6205.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 6168.60 | 6304.95 | 6205.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 6140.00 | 6271.96 | 6199.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:45:00 | 6123.15 | 6271.96 | 6199.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 6103.00 | 6222.39 | 6188.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:00:00 | 6103.00 | 6222.39 | 6188.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 6100.00 | 6197.91 | 6180.68 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 6034.65 | 6144.38 | 6158.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 5944.00 | 6104.30 | 6138.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 6207.30 | 6029.09 | 6068.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 6207.30 | 6029.09 | 6068.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 6207.30 | 6029.09 | 6068.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 6233.40 | 6029.09 | 6068.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 6184.60 | 6060.19 | 6078.74 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 6170.25 | 6105.61 | 6097.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 6282.80 | 6158.10 | 6123.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 6196.45 | 6203.37 | 6156.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:00:00 | 6196.45 | 6203.37 | 6156.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 6157.05 | 6194.10 | 6156.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 6164.05 | 6194.10 | 6156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 6175.80 | 6190.44 | 6158.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:15:00 | 6150.35 | 6190.44 | 6158.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 6086.05 | 6169.56 | 6151.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 6086.05 | 6169.56 | 6151.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 6070.00 | 6149.65 | 6144.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 6070.00 | 6149.65 | 6144.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 6055.90 | 6130.90 | 6136.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 12:15:00 | 6006.45 | 6086.30 | 6111.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 6079.50 | 6063.60 | 6090.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 6079.50 | 6063.60 | 6090.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 6079.50 | 6063.60 | 6090.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 6020.75 | 6056.91 | 6082.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 5719.71 | 5970.63 | 6024.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 6046.65 | 5985.84 | 6026.20 | SL hit (close>ema200) qty=0.50 sl=5985.84 alert=retest2 |

### Cycle 143 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 6231.40 | 6070.50 | 6052.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 6255.00 | 6193.52 | 6145.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 14:15:00 | 6223.80 | 6225.60 | 6182.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 15:00:00 | 6223.80 | 6225.60 | 6182.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 6214.35 | 6236.77 | 6205.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 6214.35 | 6236.77 | 6205.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 6207.50 | 6230.92 | 6205.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 6207.50 | 6230.92 | 6205.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 6254.65 | 6235.66 | 6209.87 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 6031.40 | 6166.10 | 6184.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 6002.10 | 6133.30 | 6167.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 5530.00 | 5509.09 | 5583.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 5530.00 | 5509.09 | 5583.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 5666.30 | 5545.48 | 5587.76 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 5729.00 | 5611.50 | 5609.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 5777.20 | 5644.64 | 5625.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 5883.70 | 5885.84 | 5816.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 5883.70 | 5885.84 | 5816.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 5754.20 | 5873.90 | 5834.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 5754.20 | 5873.90 | 5834.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 5725.80 | 5844.28 | 5824.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 5725.80 | 5844.28 | 5824.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 5750.25 | 5806.32 | 5809.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 5718.80 | 5788.82 | 5801.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 5485.55 | 5480.32 | 5543.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 5485.55 | 5480.32 | 5543.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 5265.00 | 5275.79 | 5380.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 5214.50 | 5275.79 | 5380.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:45:00 | 5244.65 | 5227.29 | 5310.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 5138.70 | 5254.55 | 5309.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 5447.80 | 5295.59 | 5282.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 5447.80 | 5295.59 | 5282.67 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 10:15:00 | 5290.00 | 5331.54 | 5332.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 5221.60 | 5309.55 | 5322.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 5211.35 | 5206.33 | 5247.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 5211.35 | 5206.33 | 5247.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 5211.35 | 5206.33 | 5247.75 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 5222.40 | 5144.96 | 5144.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5311.55 | 5210.47 | 5179.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 11:15:00 | 5200.70 | 5224.88 | 5198.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 11:15:00 | 5200.70 | 5224.88 | 5198.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 5200.70 | 5224.88 | 5198.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 5200.70 | 5224.88 | 5198.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 5225.75 | 5225.05 | 5200.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 13:15:00 | 5237.40 | 5225.05 | 5200.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 5192.95 | 5244.23 | 5230.17 | SL hit (close<static) qty=1.00 sl=5199.10 alert=retest2 |

### Cycle 150 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 5369.20 | 5520.63 | 5540.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 5308.80 | 5478.26 | 5519.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 5305.00 | 5291.58 | 5374.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 5305.00 | 5291.58 | 5374.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 5322.10 | 5293.95 | 5354.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:30:00 | 5349.30 | 5293.95 | 5354.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4628.50 | 4552.07 | 4661.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 4610.00 | 4567.86 | 4658.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 4600.40 | 4578.29 | 4655.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:30:00 | 4605.10 | 4598.72 | 4651.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 14:30:00 | 4608.80 | 4597.41 | 4646.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 4379.50 | 4450.51 | 4530.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 4370.38 | 4450.51 | 4530.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 4374.85 | 4450.51 | 4530.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 4378.36 | 4450.51 | 4530.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 4480.45 | 4456.50 | 4526.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 4480.45 | 4456.50 | 4526.34 | SL hit (close>ema200) qty=0.50 sl=4456.50 alert=retest2 |

### Cycle 151 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 4642.00 | 4556.20 | 4547.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 4678.00 | 4604.17 | 4573.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 4623.00 | 4673.45 | 4644.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 4623.00 | 4673.45 | 4644.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 4623.00 | 4673.45 | 4644.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 4705.00 | 4670.06 | 4645.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 11:15:00 | 5175.50 | 5056.76 | 4953.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 5406.50 | 5444.42 | 5446.69 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 5465.50 | 5450.73 | 5449.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 14:15:00 | 5504.00 | 5469.84 | 5458.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 5510.50 | 5518.32 | 5492.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 5510.50 | 5518.32 | 5492.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 5475.50 | 5509.76 | 5491.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 5457.00 | 5509.76 | 5491.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 5453.00 | 5498.41 | 5487.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 5426.00 | 5498.41 | 5487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 5415.00 | 5471.66 | 5476.67 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 5707.50 | 5498.53 | 5482.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 5802.00 | 5628.54 | 5551.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 5791.00 | 5792.57 | 5701.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 5791.00 | 5792.57 | 5701.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 5700.00 | 5753.71 | 5715.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 5700.00 | 5753.71 | 5715.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 5697.00 | 5742.37 | 5713.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 5643.50 | 5742.37 | 5713.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5756.50 | 5740.58 | 5717.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 5768.00 | 5740.58 | 5717.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 5584.50 | 5713.75 | 5709.70 | SL hit (close<static) qty=1.00 sl=5710.50 alert=retest2 |

### Cycle 156 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 5585.50 | 5688.10 | 5698.41 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 5693.00 | 5678.15 | 5678.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 5714.00 | 5685.32 | 5681.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 5679.50 | 5688.90 | 5683.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 5700.00 | 5691.12 | 5685.40 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 5665.50 | 5681.82 | 5682.26 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5698.35 | 5689.74 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 5670.00 | 5690.86 | 5691.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 5566.00 | 5647.41 | 5667.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5710.00 | 5627.79 | 5643.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5697.50 | 5641.73 | 5648.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 5680.00 | 5647.18 | 5650.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 5687.00 | 5655.15 | 5653.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 5687.00 | 5655.15 | 5653.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 5690.00 | 5673.56 | 5665.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 5664.50 | 5674.29 | 5668.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 5628.00 | 5665.03 | 5665.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 5612.00 | 5637.22 | 5648.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 5667.00 | 5649.42 | 5647.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 5677.00 | 5654.94 | 5650.16 | Break + close above crossover candle high |

### Cycle 164 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 5606.50 | 5645.25 | 5646.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 5602.50 | 5631.86 | 5639.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 5635.50 | 5626.92 | 5635.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 5625.00 | 5626.53 | 5634.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 5553.50 | 5626.53 | 5634.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 5633.00 | 5539.11 | 5532.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5633.00 | 5539.11 | 5532.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 5665.00 | 5616.65 | 5585.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 5847.50 | 5937.02 | 5885.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 5911.50 | 5931.92 | 5888.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 5934.00 | 5931.92 | 5888.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 5919.00 | 5939.71 | 5903.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5940.00 | 5925.40 | 5905.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 5920.50 | 5924.22 | 5908.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 5906.50 | 5920.67 | 5908.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:00:00 | 5906.50 | 5920.67 | 5908.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 5893.50 | 5915.24 | 5906.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 5882.50 | 5915.24 | 5906.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 5883.00 | 5908.79 | 5904.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 5861.50 | 5908.79 | 5904.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 5943.00 | 5912.70 | 5907.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 5964.00 | 5912.70 | 5907.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 5877.00 | 5992.60 | 6005.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 5877.00 | 5992.60 | 6005.58 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 6034.00 | 5989.57 | 5984.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 6106.00 | 6012.86 | 5995.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 6081.00 | 6086.21 | 6049.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 6085.00 | 6086.21 | 6049.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 6078.50 | 6100.93 | 6077.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 6078.50 | 6100.93 | 6077.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 6080.00 | 6096.74 | 6077.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 6142.00 | 6096.74 | 6077.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 6109.00 | 6112.49 | 6092.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 6043.50 | 6097.35 | 6091.63 | SL hit (close<static) qty=1.00 sl=6075.50 alert=retest2 |

### Cycle 168 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 6044.50 | 6081.28 | 6084.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 13:15:00 | 6031.50 | 6060.09 | 6070.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 6032.50 | 6031.99 | 6050.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 6032.50 | 6031.99 | 6050.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 6070.00 | 6040.31 | 6051.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 6070.00 | 6040.31 | 6051.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 6040.00 | 6040.25 | 6050.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 6066.00 | 6040.25 | 6050.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 6042.00 | 6040.60 | 6049.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 6057.00 | 6040.60 | 6049.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 6042.00 | 6040.88 | 6048.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 6078.50 | 6040.88 | 6048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 6046.00 | 6041.90 | 6048.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 6060.00 | 6041.90 | 6048.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6029.50 | 6039.42 | 6046.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 6044.00 | 6039.42 | 6046.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 5998.00 | 6031.14 | 6042.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:30:00 | 6027.00 | 6031.14 | 6042.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 6035.00 | 6030.29 | 6039.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 6035.00 | 6030.29 | 6039.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 6028.00 | 6029.83 | 6038.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 6076.50 | 6029.83 | 6038.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 6076.50 | 6039.16 | 6042.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 6093.50 | 6039.16 | 6042.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 6014.50 | 6034.23 | 6039.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 5990.00 | 6033.19 | 6038.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 5991.00 | 6014.22 | 6025.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 5986.50 | 6014.22 | 6025.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5690.50 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5691.45 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5687.18 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 5612.00 | 5589.59 | 5628.58 | SL hit (close>ema200) qty=0.50 sl=5589.59 alert=retest2 |

### Cycle 169 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 5650.00 | 5555.73 | 5551.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 5683.00 | 5581.18 | 5563.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 5748.50 | 5749.57 | 5687.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 5748.50 | 5749.57 | 5687.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 5684.00 | 5729.01 | 5692.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 5684.00 | 5729.01 | 5692.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 5646.00 | 5712.41 | 5688.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:15:00 | 5639.00 | 5712.41 | 5688.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 5599.50 | 5662.83 | 5669.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5105.00 | 5534.03 | 5607.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 5158.00 | 5151.60 | 5228.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 5158.00 | 5151.60 | 5228.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 5178.00 | 5160.28 | 5194.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 5186.00 | 5160.28 | 5194.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 5135.50 | 5158.48 | 5188.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 5101.50 | 5148.38 | 5180.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 5109.00 | 5146.54 | 5167.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 5103.50 | 5146.54 | 5167.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 5097.50 | 5137.33 | 5151.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5116.50 | 5084.03 | 5108.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 5116.50 | 5084.03 | 5108.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5157.50 | 5098.73 | 5113.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 5157.50 | 5098.73 | 5113.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5175.50 | 5114.08 | 5118.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 5183.50 | 5114.08 | 5118.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 5175.00 | 5126.26 | 5123.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 5175.00 | 5126.26 | 5123.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 5197.50 | 5154.34 | 5139.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 5136.00 | 5154.35 | 5142.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 5076.00 | 5138.68 | 5136.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 5076.00 | 5138.68 | 5136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 5093.00 | 5129.55 | 5132.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 5047.00 | 5099.58 | 5117.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5124.50 | 5096.39 | 5112.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5110.50 | 5099.22 | 5111.97 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 5151.50 | 5122.41 | 5120.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 5175.00 | 5132.93 | 5125.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 5139.50 | 5142.57 | 5131.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 5142.00 | 5142.46 | 5132.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 5135.00 | 5142.46 | 5132.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 5158.50 | 5145.67 | 5135.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 5144.00 | 5145.67 | 5135.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 5138.00 | 5144.13 | 5135.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 5138.00 | 5144.13 | 5135.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 5119.50 | 5139.21 | 5133.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 5119.50 | 5139.21 | 5133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 5105.00 | 5132.37 | 5131.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 5105.00 | 5132.37 | 5131.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 5100.00 | 5125.89 | 5128.38 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 5156.00 | 5131.93 | 5130.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 5173.00 | 5140.14 | 5134.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 5220.00 | 5225.83 | 5200.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 5220.00 | 5225.83 | 5200.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 5227.50 | 5226.17 | 5203.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 5222.50 | 5226.17 | 5203.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5257.50 | 5278.58 | 5255.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 5264.00 | 5278.58 | 5255.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 5277.00 | 5278.26 | 5257.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 5270.00 | 5278.26 | 5257.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 5234.00 | 5269.41 | 5255.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 5234.00 | 5269.41 | 5255.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 5250.00 | 5265.53 | 5254.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 5237.00 | 5265.53 | 5254.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 5252.00 | 5262.82 | 5254.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:30:00 | 5246.50 | 5262.82 | 5254.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 5231.00 | 5250.57 | 5250.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 5255.50 | 5250.57 | 5250.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5201.50 | 5240.75 | 5245.69 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 5276.50 | 5245.01 | 5242.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 5346.00 | 5265.21 | 5251.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 5378.50 | 5347.40 | 5318.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 5376.00 | 5349.52 | 5322.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:45:00 | 5375.00 | 5353.41 | 5326.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 5383.50 | 5353.41 | 5326.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 5350.00 | 5363.21 | 5340.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 5475.00 | 5363.21 | 5340.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 5335.50 | 5398.80 | 5401.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 5335.50 | 5398.80 | 5401.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 5281.00 | 5341.05 | 5368.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 5320.00 | 5319.09 | 5343.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:15:00 | 5371.00 | 5319.09 | 5343.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 5364.50 | 5328.17 | 5345.51 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 5424.00 | 5359.71 | 5357.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 5437.50 | 5406.17 | 5385.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 5387.00 | 5409.44 | 5392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 5409.50 | 5409.45 | 5394.32 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 5319.00 | 5382.78 | 5385.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 5288.00 | 5320.63 | 5343.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 5074.00 | 5069.27 | 5138.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:45:00 | 5090.00 | 5069.27 | 5138.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 5132.00 | 5089.09 | 5121.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 5132.00 | 5089.09 | 5121.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 5132.00 | 5097.67 | 5122.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 5309.00 | 5097.67 | 5122.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 5369.50 | 5152.04 | 5145.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 5386.50 | 5198.93 | 5167.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 5329.50 | 5336.19 | 5263.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:30:00 | 5359.00 | 5336.19 | 5263.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 5344.00 | 5381.27 | 5352.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 5344.00 | 5381.27 | 5352.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 5356.00 | 5376.22 | 5352.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 5352.00 | 5376.22 | 5352.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5380.50 | 5377.07 | 5355.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 5391.00 | 5373.81 | 5361.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5390.00 | 5373.99 | 5364.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 5280.00 | 5473.85 | 5494.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 5280.00 | 5473.85 | 5494.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 5140.50 | 5233.57 | 5302.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 5222.00 | 5197.67 | 5244.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:30:00 | 5226.50 | 5197.67 | 5244.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 4950.50 | 4867.32 | 4910.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 4950.50 | 4867.32 | 4910.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 4934.80 | 4880.81 | 4912.89 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 5040.00 | 4942.14 | 4934.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 5199.00 | 5058.43 | 5003.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 5254.80 | 5268.10 | 5217.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 5254.80 | 5268.10 | 5217.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 5262.20 | 5337.37 | 5313.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 5262.20 | 5337.37 | 5313.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 5220.40 | 5313.98 | 5305.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 5220.40 | 5313.98 | 5305.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 5237.40 | 5298.66 | 5298.88 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 5326.70 | 5299.51 | 5298.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 5354.60 | 5315.57 | 5306.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5676.80 | 5338.88 | 5326.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 5816.60 | 5849.54 | 5849.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 5816.60 | 5849.54 | 5849.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 5805.30 | 5835.49 | 5843.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 5850.10 | 5835.79 | 5840.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 5849.70 | 5838.57 | 5841.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 5849.70 | 5838.57 | 5841.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 5863.60 | 5843.58 | 5843.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 5879.10 | 5854.91 | 5848.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 5907.20 | 5919.84 | 5893.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 10:00:00 | 5907.20 | 5919.84 | 5893.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5887.70 | 5913.41 | 5893.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 5887.70 | 5913.41 | 5893.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 5878.00 | 5906.33 | 5891.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 5878.00 | 5906.33 | 5891.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 5885.90 | 5902.24 | 5891.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 5915.20 | 5904.83 | 5893.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 5856.00 | 5898.49 | 5893.75 | SL hit (close<static) qty=1.00 sl=5875.10 alert=retest2 |

### Cycle 188 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 5839.50 | 5891.43 | 5895.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 5816.00 | 5853.73 | 5872.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 5869.00 | 5853.07 | 5868.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 5842.50 | 5850.96 | 5866.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 5831.50 | 5849.46 | 5863.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 5864.00 | 5829.88 | 5827.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5864.00 | 5829.88 | 5827.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 6014.50 | 5878.93 | 5851.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 6131.50 | 6138.83 | 6080.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 6131.50 | 6138.83 | 6080.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 6096.50 | 6128.95 | 6085.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 6090.50 | 6128.95 | 6085.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 6103.50 | 6123.86 | 6087.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 6093.00 | 6123.86 | 6087.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 6040.50 | 6103.37 | 6084.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 6040.50 | 6103.37 | 6084.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 6049.00 | 6092.50 | 6081.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 6063.50 | 6095.60 | 6083.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 6108.00 | 6095.60 | 6083.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 6065.00 | 6094.50 | 6095.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 6065.00 | 6094.50 | 6095.94 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 6281.00 | 6131.80 | 6112.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 6318.00 | 6191.47 | 6144.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 6338.00 | 6349.02 | 6285.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 6348.00 | 6349.02 | 6285.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 6285.50 | 6336.31 | 6285.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 6296.50 | 6336.31 | 6285.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6300.00 | 6329.05 | 6287.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 6291.50 | 6329.05 | 6287.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 6299.00 | 6323.04 | 6288.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 6279.50 | 6323.04 | 6288.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 6301.00 | 6315.29 | 6292.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 6301.00 | 6315.29 | 6292.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 6278.00 | 6307.83 | 6291.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 6389.00 | 6307.83 | 6291.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 6370.00 | 6391.50 | 6392.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 6370.00 | 6391.50 | 6392.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 6346.00 | 6370.48 | 6380.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 6395.00 | 6375.39 | 6382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 6383.00 | 6376.91 | 6382.24 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 6403.00 | 6388.49 | 6386.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 6425.00 | 6395.79 | 6390.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 6369.50 | 6390.53 | 6388.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 6369.50 | 6386.33 | 6386.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 6342.50 | 6377.56 | 6382.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 6393.50 | 6376.66 | 6381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 6378.50 | 6377.03 | 6380.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 6371.00 | 6377.03 | 6380.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 6348.50 | 6371.32 | 6377.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 6316.50 | 6356.86 | 6370.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 6321.50 | 6347.53 | 6363.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 6321.50 | 6343.72 | 6360.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 6440.00 | 6360.95 | 6363.95 | SL hit (close>static) qty=1.00 sl=6408.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 6454.50 | 6379.66 | 6372.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 6482.50 | 6400.23 | 6382.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 6431.00 | 6484.28 | 6455.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 6396.50 | 6466.72 | 6449.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 6400.00 | 6466.72 | 6449.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 6405.00 | 6438.18 | 6438.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 6340.00 | 6418.55 | 6429.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 13:15:00 | 6132.00 | 6123.98 | 6195.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:30:00 | 6132.50 | 6123.98 | 6195.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 6196.00 | 6138.38 | 6195.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 6196.00 | 6138.38 | 6195.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 6225.50 | 6155.80 | 6198.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 6287.00 | 6155.80 | 6198.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 6266.50 | 6177.94 | 6204.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 6271.00 | 6177.94 | 6204.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 6269.50 | 6196.25 | 6210.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 6263.50 | 6196.25 | 6210.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 6249.00 | 6219.16 | 6219.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 6344.50 | 6252.20 | 6234.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 6293.50 | 6295.99 | 6268.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 6293.50 | 6295.99 | 6268.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 6240.00 | 6284.28 | 6270.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 6276.50 | 6265.26 | 6263.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 6232.00 | 6258.61 | 6260.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 6232.00 | 6258.61 | 6260.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 6222.00 | 6251.29 | 6257.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 6316.50 | 6249.26 | 6255.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 6304.50 | 6260.31 | 6259.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 6349.00 | 6321.45 | 6303.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 6507.50 | 6507.78 | 6435.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 6495.50 | 6503.78 | 6461.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 6495.50 | 6503.78 | 6461.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 6459.50 | 6503.78 | 6461.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 6442.50 | 6486.60 | 6460.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 6437.50 | 6486.60 | 6460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 6420.00 | 6473.28 | 6456.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 6420.00 | 6473.28 | 6456.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 6396.00 | 6444.94 | 6446.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 6358.00 | 6427.55 | 6438.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 6285.00 | 6255.45 | 6253.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 6315.50 | 6273.52 | 6262.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 6211.50 | 6273.00 | 6268.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 6188.00 | 6256.00 | 6261.52 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 6388.00 | 6271.74 | 6256.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 6445.50 | 6306.49 | 6274.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 6448.00 | 6454.97 | 6382.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 6448.00 | 6454.97 | 6382.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 6410.00 | 6444.28 | 6410.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 6402.00 | 6444.28 | 6410.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 6440.00 | 6443.42 | 6413.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 6450.00 | 6443.42 | 6413.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 6412.50 | 6437.24 | 6413.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 6402.00 | 6437.24 | 6413.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 6404.00 | 6430.59 | 6412.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 6405.00 | 6430.59 | 6412.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 6423.50 | 6429.17 | 6413.45 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 6344.50 | 6400.85 | 6404.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 12:15:00 | 6321.00 | 6376.82 | 6392.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 6349.00 | 6342.89 | 6366.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:30:00 | 6313.50 | 6342.89 | 6366.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 6359.50 | 6340.50 | 6357.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 6359.50 | 6340.50 | 6357.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 6341.50 | 6340.70 | 6355.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 6329.00 | 6340.70 | 6355.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 6412.00 | 6320.72 | 6330.46 | SL hit (close>static) qty=1.00 sl=6360.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 6406.50 | 6337.88 | 6337.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 6475.50 | 6365.40 | 6349.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 6361.50 | 6418.14 | 6398.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 6393.50 | 6413.21 | 6397.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 6415.00 | 6413.21 | 6397.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 6374.50 | 6405.47 | 6395.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 6374.50 | 6405.47 | 6395.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 6399.00 | 6404.17 | 6396.12 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 6340.00 | 6384.27 | 6388.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 6120.00 | 6322.73 | 6358.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 6239.00 | 6221.36 | 6284.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 6239.00 | 6221.36 | 6284.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 6365.00 | 6249.87 | 6286.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 6353.50 | 6249.87 | 6286.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 6235.50 | 6248.62 | 6279.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 6175.00 | 6248.73 | 6268.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 6171.00 | 6237.68 | 6261.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 6153.00 | 6229.25 | 6255.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 6172.00 | 6178.96 | 6219.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 6213.00 | 6157.41 | 6190.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 6213.00 | 6157.41 | 6190.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 6238.50 | 6173.63 | 6194.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 6278.00 | 6193.30 | 6201.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 6260.50 | 6206.74 | 6206.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 6242.50 | 6213.89 | 6210.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 6242.50 | 6213.89 | 6210.13 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 6027.00 | 6172.40 | 6191.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 6010.00 | 6068.09 | 6118.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 6062.50 | 6031.42 | 6067.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:00:00 | 6062.50 | 6031.42 | 6067.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 6055.00 | 6036.14 | 6066.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 6055.00 | 6036.14 | 6066.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 6157.00 | 6060.31 | 6074.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 6157.00 | 6060.31 | 6074.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 6110.00 | 6070.25 | 6078.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 6055.00 | 6078.70 | 6081.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 6079.00 | 6074.97 | 6078.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 6024.00 | 6062.47 | 6072.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 6279.00 | 6103.11 | 6085.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6279.00 | 6103.11 | 6085.31 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 5845.00 | 6108.25 | 6122.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 5827.00 | 5945.42 | 5994.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 5857.00 | 5854.35 | 5922.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 5857.00 | 5854.35 | 5922.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 5856.50 | 5852.48 | 5909.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 5842.00 | 5852.48 | 5909.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 5940.00 | 5885.23 | 5895.08 | SL hit (close>static) qty=1.00 sl=5929.50 alert=retest2 |

### Cycle 211 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 5697.00 | 5561.62 | 5555.97 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 5461.00 | 5565.02 | 5573.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 5412.50 | 5505.93 | 5537.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 4795.50 | 4747.58 | 4871.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:45:00 | 4794.00 | 4747.58 | 4871.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 4882.00 | 4771.76 | 4835.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 4878.00 | 4771.76 | 4835.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 4893.50 | 4796.11 | 4841.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 4887.50 | 4796.11 | 4841.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 4776.50 | 4796.57 | 4833.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 4769.00 | 4790.36 | 4824.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:45:00 | 4768.50 | 4780.85 | 4806.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4530.55 | 4757.18 | 4785.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 4771.40 | 4757.18 | 4785.08 | SL hit (close>static) qty=0.50 sl=4757.18 alert=retest2 |

### Cycle 213 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 4771.20 | 4707.24 | 4703.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 10:15:00 | 4806.00 | 4763.73 | 4737.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 12:15:00 | 4755.40 | 4766.52 | 4743.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 13:00:00 | 4755.40 | 4766.52 | 4743.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 4787.00 | 4774.76 | 4755.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 4772.00 | 4774.76 | 4755.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 4757.90 | 4775.46 | 4759.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 4757.90 | 4775.46 | 4759.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 4777.00 | 4775.77 | 4760.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 4789.80 | 4777.93 | 4763.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:30:00 | 4807.10 | 4786.25 | 4768.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:00:00 | 4808.00 | 4799.42 | 4777.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 4795.40 | 4799.33 | 4781.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4776.00 | 4794.67 | 4781.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 4776.00 | 4794.67 | 4781.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4770.00 | 4789.73 | 4780.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 4770.00 | 4789.73 | 4780.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 4734.00 | 4778.59 | 4775.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 4734.00 | 4778.59 | 4775.93 | SL hit (close<static) qty=1.00 sl=4750.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4747.90 | 4772.45 | 4773.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4644.60 | 4746.88 | 4761.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 4728.00 | 4717.87 | 4740.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 4715.00 | 4717.30 | 4738.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 4620.00 | 4719.84 | 4737.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 15:15:00 | 4848.90 | 4664.08 | 4667.77 | SL hit (close>static) qty=1.00 sl=4739.60 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 4753.80 | 4625.77 | 4625.73 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4600.00 | 4646.36 | 4646.87 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 4729.10 | 4657.53 | 4648.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 4739.00 | 4700.26 | 4675.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 4758.20 | 4730.81 | 4707.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 4790.50 | 4732.81 | 4712.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-02 13:15:00 | 5234.02 | 5105.44 | 5030.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 5442.00 | 5456.25 | 5457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 11:15:00 | 5419.40 | 5449.48 | 5453.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 15:15:00 | 5369.00 | 5348.44 | 5377.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:15:00 | 5117.30 | 5348.44 | 5377.96 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 4861.44 | 4990.23 | 5088.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 4820.00 | 4819.54 | 4906.54 | SL hit (close>ema200) qty=0.50 sl=4819.54 alert=retest1 |

### Cycle 219 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 4836.70 | 4810.14 | 4808.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 4949.80 | 4841.46 | 4823.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 4957.00 | 4959.06 | 4905.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 4957.00 | 4959.06 | 4905.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 4953.20 | 4953.67 | 4912.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 4978.10 | 4959.19 | 4918.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 14:45:00 | 3836.55 | 2024-04-22 10:15:00 | 3644.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-19 09:15:00 | 3806.60 | 2024-04-22 10:15:00 | 3616.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-19 10:45:00 | 3835.95 | 2024-04-22 10:15:00 | 3644.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-22 09:15:00 | 3757.55 | 2024-04-22 10:15:00 | 3569.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 14:45:00 | 3836.55 | 2024-04-22 14:15:00 | 3452.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-19 09:15:00 | 3806.60 | 2024-04-22 14:15:00 | 3425.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-19 10:45:00 | 3835.95 | 2024-04-22 14:15:00 | 3452.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-22 09:15:00 | 3757.55 | 2024-04-22 14:15:00 | 3381.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-22 10:15:00 | 3624.65 | 2024-04-22 14:15:00 | 3443.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-22 10:15:00 | 3624.65 | 2024-04-26 09:15:00 | 3482.40 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-05-13 09:30:00 | 3322.00 | 2024-05-13 14:15:00 | 3392.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-05-22 09:15:00 | 3523.35 | 2024-05-22 14:15:00 | 3498.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-22 09:45:00 | 3530.00 | 2024-05-22 14:15:00 | 3498.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-05-22 10:45:00 | 3524.00 | 2024-05-22 14:15:00 | 3498.25 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-05-23 09:15:00 | 3522.05 | 2024-05-30 09:15:00 | 3581.15 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-05-27 09:15:00 | 3593.95 | 2024-05-30 09:15:00 | 3581.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-06-10 15:00:00 | 3750.00 | 2024-06-28 11:15:00 | 4125.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-18 11:30:00 | 4836.90 | 2024-07-19 09:15:00 | 4654.55 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-07-18 12:30:00 | 4851.35 | 2024-07-19 09:15:00 | 4654.55 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2024-08-06 15:00:00 | 4497.00 | 2024-08-07 12:15:00 | 4648.40 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-08-12 13:15:00 | 4735.00 | 2024-08-23 09:15:00 | 4875.50 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest2 | 2024-08-13 09:15:00 | 4744.00 | 2024-08-23 09:15:00 | 4875.50 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2024-08-13 13:15:00 | 4714.60 | 2024-08-23 09:15:00 | 4875.50 | STOP_HIT | 1.00 | 3.41% |
| BUY | retest2 | 2024-08-13 15:15:00 | 4710.00 | 2024-08-23 09:15:00 | 4875.50 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest2 | 2024-08-21 10:45:00 | 4944.30 | 2024-08-23 12:15:00 | 4902.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-21 11:15:00 | 4935.20 | 2024-08-23 12:15:00 | 4902.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-21 12:00:00 | 4938.80 | 2024-08-23 12:15:00 | 4902.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-08-22 09:15:00 | 4935.85 | 2024-08-23 12:15:00 | 4902.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-08-23 11:30:00 | 4901.75 | 2024-08-23 12:15:00 | 4902.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-08-29 10:15:00 | 5039.90 | 2024-09-06 14:15:00 | 5186.05 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2024-08-29 13:15:00 | 5042.55 | 2024-09-06 14:15:00 | 5186.05 | STOP_HIT | 1.00 | 2.85% |
| BUY | retest2 | 2024-09-12 14:00:00 | 5302.25 | 2024-09-18 09:15:00 | 5158.75 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-09-12 14:30:00 | 5298.50 | 2024-09-18 09:15:00 | 5158.75 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-09-13 09:15:00 | 5301.95 | 2024-09-18 09:15:00 | 5158.75 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-09-24 14:15:00 | 5385.55 | 2024-09-25 14:15:00 | 5320.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-10-04 12:15:00 | 5199.00 | 2024-10-08 14:15:00 | 5291.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-10-07 10:15:00 | 5197.55 | 2024-10-08 14:15:00 | 5291.20 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-10-08 09:15:00 | 5195.45 | 2024-10-08 14:15:00 | 5291.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-10-16 14:30:00 | 5600.35 | 2024-10-17 09:15:00 | 5490.65 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-11-11 11:00:00 | 5728.55 | 2024-11-13 09:15:00 | 5613.55 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-11-11 12:00:00 | 5731.00 | 2024-11-13 09:15:00 | 5613.55 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-11-11 13:30:00 | 5732.05 | 2024-11-13 09:15:00 | 5613.55 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-11-12 09:45:00 | 5729.10 | 2024-11-13 09:15:00 | 5613.55 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-12-03 11:30:00 | 5941.85 | 2024-12-17 09:15:00 | 6536.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 12:30:00 | 5935.00 | 2024-12-17 09:15:00 | 6528.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 13:30:00 | 5939.65 | 2024-12-17 09:15:00 | 6533.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-27 11:15:00 | 6364.50 | 2024-12-30 10:15:00 | 6415.85 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-30 09:15:00 | 6340.85 | 2024-12-30 10:15:00 | 6415.85 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-12-31 11:15:00 | 6363.00 | 2025-01-01 11:15:00 | 6369.15 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-12-31 11:45:00 | 6405.10 | 2025-01-01 11:15:00 | 6369.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-01-07 13:45:00 | 6340.00 | 2025-01-10 11:15:00 | 6398.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-01-07 15:00:00 | 6326.35 | 2025-01-10 11:15:00 | 6398.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-02-01 11:45:00 | 6020.75 | 2025-02-03 09:15:00 | 5719.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 6020.75 | 2025-02-03 10:15:00 | 6046.65 | STOP_HIT | 0.50 | -0.43% |
| SELL | retest2 | 2025-02-03 11:15:00 | 6030.15 | 2025-02-04 09:15:00 | 6231.40 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-02-03 12:30:00 | 6017.80 | 2025-02-04 09:15:00 | 6231.40 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-02-03 13:15:00 | 6026.35 | 2025-02-04 09:15:00 | 6231.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-03-03 09:15:00 | 5214.50 | 2025-03-05 11:15:00 | 5447.80 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-03-03 13:45:00 | 5244.65 | 2025-03-05 11:15:00 | 5447.80 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-03-04 09:15:00 | 5138.70 | 2025-03-05 11:15:00 | 5447.80 | STOP_HIT | 1.00 | -6.02% |
| BUY | retest2 | 2025-03-19 13:15:00 | 5237.40 | 2025-03-20 14:15:00 | 5192.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-03-21 09:30:00 | 5273.35 | 2025-04-01 09:15:00 | 5369.20 | STOP_HIT | 1.00 | 1.82% |
| SELL | retest2 | 2025-04-08 10:30:00 | 4610.00 | 2025-04-09 15:15:00 | 4379.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 11:30:00 | 4600.40 | 2025-04-09 15:15:00 | 4370.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 13:30:00 | 4605.10 | 2025-04-09 15:15:00 | 4374.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 14:30:00 | 4608.80 | 2025-04-09 15:15:00 | 4378.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 4610.00 | 2025-04-11 09:15:00 | 4480.45 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2025-04-08 11:30:00 | 4600.40 | 2025-04-11 09:15:00 | 4480.45 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-04-08 13:30:00 | 4605.10 | 2025-04-11 09:15:00 | 4480.45 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-04-08 14:30:00 | 4608.80 | 2025-04-11 09:15:00 | 4480.45 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2025-04-17 11:15:00 | 4705.00 | 2025-04-23 11:15:00 | 5175.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 15:15:00 | 5768.00 | 2025-05-15 09:15:00 | 5584.50 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-05-23 11:30:00 | 5680.00 | 2025-05-23 12:15:00 | 5687.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-06-02 09:15:00 | 5553.50 | 2025-06-05 11:15:00 | 5633.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-12 11:15:00 | 5934.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-12 14:15:00 | 5919.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5940.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-13 12:15:00 | 5920.50 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-16 11:15:00 | 5964.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-25 09:15:00 | 6142.00 | 2025-06-26 09:15:00 | 6043.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-25 12:45:00 | 6109.00 | 2025-06-26 09:15:00 | 6043.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-02 13:15:00 | 5990.00 | 2025-07-10 09:15:00 | 5690.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 11:30:00 | 5991.00 | 2025-07-10 09:15:00 | 5691.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 12:00:00 | 5986.50 | 2025-07-10 09:15:00 | 5687.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-02 13:15:00 | 5990.00 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2025-07-03 11:30:00 | 5991.00 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-07-03 12:00:00 | 5986.50 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.26% |
| SELL | retest2 | 2025-07-30 10:30:00 | 5101.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-31 09:30:00 | 5109.00 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 10:00:00 | 5103.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-01 11:30:00 | 5097.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-22 09:30:00 | 5378.50 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-22 10:30:00 | 5376.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-08-22 11:45:00 | 5375.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-22 12:15:00 | 5383.50 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-08-25 09:15:00 | 5475.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-09-16 09:45:00 | 5391.00 | 2025-09-22 09:15:00 | 5280.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-09-16 14:00:00 | 5390.00 | 2025-09-22 09:15:00 | 5280.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5676.80 | 2025-10-28 11:15:00 | 5816.60 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-10-31 14:00:00 | 5915.20 | 2025-11-03 09:15:00 | 5856.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-03 11:00:00 | 5899.50 | 2025-11-04 10:15:00 | 5874.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-11-03 12:30:00 | 5911.00 | 2025-11-04 10:15:00 | 5874.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-06 15:15:00 | 5831.50 | 2025-11-10 13:15:00 | 5864.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-14 14:30:00 | 6063.50 | 2025-11-18 15:15:00 | 6065.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-14 15:00:00 | 6108.00 | 2025-11-18 15:15:00 | 6065.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-24 09:15:00 | 6389.00 | 2025-11-28 13:15:00 | 6370.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-03 10:30:00 | 6316.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-12-03 12:30:00 | 6321.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-03 14:15:00 | 6321.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-16 12:15:00 | 6276.50 | 2025-12-16 12:15:00 | 6232.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-14 09:15:00 | 6329.00 | 2026-01-16 09:15:00 | 6412.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-23 10:45:00 | 6175.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-23 11:30:00 | 6171.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-23 13:15:00 | 6153.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-27 09:45:00 | 6172.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-01 15:15:00 | 6055.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-02-02 10:00:00 | 6079.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-02-02 10:45:00 | 6024.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2026-02-09 10:15:00 | 5842.00 | 2026-02-10 11:15:00 | 5940.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-11 09:15:00 | 5839.00 | 2026-02-12 09:15:00 | 5547.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 5839.00 | 2026-02-13 09:15:00 | 5255.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 4769.00 | 2026-03-02 09:15:00 | 4530.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 4769.00 | 2026-03-02 09:15:00 | 4771.40 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2026-02-27 12:45:00 | 4768.50 | 2026-03-02 09:15:00 | 4530.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:45:00 | 4768.50 | 2026-03-02 09:15:00 | 4771.40 | STOP_HIT | 0.50 | -0.06% |
| SELL | retest2 | 2026-03-02 10:15:00 | 4760.00 | 2026-03-06 09:15:00 | 4754.30 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-03-02 11:00:00 | 4722.70 | 2026-03-06 11:15:00 | 4771.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-03-05 09:30:00 | 4693.60 | 2026-03-06 11:15:00 | 4771.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-03-10 13:45:00 | 4789.80 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-10 14:30:00 | 4807.10 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-03-11 10:00:00 | 4808.00 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-11 12:00:00 | 4795.40 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-03-13 09:15:00 | 4620.00 | 2026-03-16 15:15:00 | 4848.90 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-03-17 09:15:00 | 4557.90 | 2026-03-18 10:15:00 | 4753.80 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-03-24 09:15:00 | 4758.20 | 2026-04-02 13:15:00 | 5234.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:15:00 | 4790.50 | 2026-04-06 09:15:00 | 5269.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-24 11:15:00 | 4861.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-27 14:15:00 | 4820.00 | STOP_HIT | 0.50 | 5.81% |
| SELL | retest2 | 2026-04-29 11:15:00 | 4814.00 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-04-29 12:00:00 | 4815.50 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-04-29 13:30:00 | 4799.00 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-05-05 10:15:00 | 4816.70 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-05 11:30:00 | 4801.30 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.74% |
