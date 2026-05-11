# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 231 |
| ALERT1 | 156 |
| ALERT2 | 156 |
| ALERT2_SKIP | 154 |
| ALERT3 | 159 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.03% / -2.28%
- **Sum % (uncompounded):** -4.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.03% | -4.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.36% | -7.1% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.95% | 3.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.36% | -7.1% |
| retest2 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.95% | 3.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-12 10:15:00 | 4635.00 | 4608.52 | 4605.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 09:15:00 | 4667.65 | 4621.13 | 4613.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-15 14:15:00 | 4621.65 | 4636.66 | 4625.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-15 14:15:00 | 4621.65 | 4636.66 | 4625.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 14:15:00 | 4621.65 | 4636.66 | 4625.96 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 4588.00 | 4618.24 | 4620.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 4555.80 | 4596.08 | 4608.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 4580.10 | 4505.98 | 4497.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 4614.80 | 4539.91 | 4514.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 4561.65 | 4629.09 | 4631.58 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 4815.00 | 4664.38 | 4646.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 4841.60 | 4788.58 | 4730.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 14:15:00 | 4945.00 | 4948.79 | 4883.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 11:15:00 | 4915.00 | 4936.02 | 4897.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 4915.00 | 4936.02 | 4897.74 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 4930.10 | 4949.36 | 4950.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 15:15:00 | 4920.00 | 4932.76 | 4940.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 4982.00 | 4951.20 | 4947.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 09:15:00 | 4993.00 | 4960.58 | 4953.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 5183.60 | 5190.65 | 5149.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 14:15:00 | 5144.05 | 5176.67 | 5150.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 5144.05 | 5176.67 | 5150.35 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 5093.95 | 5140.58 | 5146.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 5078.45 | 5114.19 | 5128.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 5040.00 | 5037.42 | 5055.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 5118.80 | 5053.70 | 5061.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 5118.80 | 5053.70 | 5061.45 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 5109.30 | 5073.72 | 5069.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 13:15:00 | 5122.90 | 5087.24 | 5076.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 5126.35 | 5128.37 | 5109.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 5108.15 | 5124.92 | 5111.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 5108.15 | 5124.92 | 5111.04 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 12:15:00 | 5081.05 | 5102.07 | 5102.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 12:15:00 | 5068.00 | 5091.32 | 5096.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 12:15:00 | 5075.75 | 5074.41 | 5083.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 13:15:00 | 5110.00 | 5081.52 | 5086.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 5110.00 | 5081.52 | 5086.20 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 15:15:00 | 5114.00 | 5092.42 | 5090.61 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 5077.80 | 5089.22 | 5089.71 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 5205.00 | 5107.37 | 5096.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 10:15:00 | 5225.00 | 5130.90 | 5108.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 5218.25 | 5226.39 | 5179.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 14:15:00 | 5135.65 | 5195.95 | 5179.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 5135.65 | 5195.95 | 5179.31 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 5160.00 | 5172.98 | 5173.64 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 5246.65 | 5187.71 | 5180.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 5184.00 | 5216.71 | 5219.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 5114.55 | 5192.00 | 5207.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 5178.60 | 5165.82 | 5185.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 5299.90 | 5195.66 | 5196.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 5299.90 | 5195.66 | 5196.15 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 5232.30 | 5202.99 | 5199.44 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 5116.10 | 5193.17 | 5200.91 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 5197.00 | 5187.74 | 5187.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 5233.80 | 5196.95 | 5191.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 5165.00 | 5194.18 | 5197.22 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 14:15:00 | 5214.95 | 5196.97 | 5196.70 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 11:15:00 | 5186.55 | 5196.72 | 5196.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 12:15:00 | 5172.85 | 5191.94 | 5194.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 5205.60 | 5191.96 | 5191.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 14:15:00 | 5220.65 | 5204.59 | 5198.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 14:15:00 | 5170.20 | 5222.31 | 5227.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 09:15:00 | 5100.15 | 5191.91 | 5212.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 09:15:00 | 4996.45 | 4988.26 | 5008.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 10:15:00 | 5006.55 | 4991.92 | 5008.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 5006.55 | 4991.92 | 5008.20 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 12:15:00 | 5042.70 | 5015.42 | 5012.71 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 4988.40 | 5010.94 | 5012.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 11:15:00 | 4960.70 | 5000.89 | 5007.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 4946.00 | 4904.96 | 4899.49 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 4878.45 | 4895.80 | 4896.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 4839.40 | 4880.35 | 4888.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 4931.85 | 4888.87 | 4883.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 4949.00 | 4916.12 | 4899.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 4919.90 | 4924.65 | 4907.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 4976.95 | 4935.08 | 4915.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 4976.95 | 4935.08 | 4915.52 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 4907.00 | 4929.66 | 4930.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 4872.95 | 4918.32 | 4925.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 4924.10 | 4900.96 | 4900.22 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 4880.80 | 4896.26 | 4898.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 4874.90 | 4891.99 | 4896.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 4950.00 | 4854.64 | 4849.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 4963.50 | 4876.41 | 4860.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 4980.00 | 5020.30 | 5023.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 4968.35 | 4998.46 | 5011.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 5053.80 | 5016.86 | 5014.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 5093.70 | 5068.06 | 5046.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 5082.65 | 5098.14 | 5072.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 5152.85 | 5109.38 | 5082.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 5152.85 | 5109.38 | 5082.21 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 4960.80 | 5066.60 | 5074.00 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 5090.00 | 5019.54 | 5017.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 5091.90 | 5034.01 | 5024.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 5021.00 | 5070.48 | 5072.21 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 5122.50 | 5081.26 | 5076.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 5141.85 | 5093.38 | 5082.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 5035.95 | 5103.79 | 5105.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 4986.25 | 5071.96 | 5089.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 5063.25 | 5054.64 | 5054.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 5136.00 | 5074.85 | 5063.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 5096.90 | 5102.87 | 5084.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 15:15:00 | 5083.00 | 5098.90 | 5084.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 5083.00 | 5098.90 | 5084.54 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 5025.00 | 5067.46 | 5072.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 4937.50 | 5041.47 | 5060.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 5009.15 | 5001.46 | 5028.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 5012.65 | 5003.70 | 5026.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 5012.65 | 5003.70 | 5026.75 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 5029.80 | 5027.35 | 5027.28 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 4998.40 | 5021.50 | 5024.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 4963.00 | 5004.89 | 5016.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 4879.60 | 4848.13 | 4846.70 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 4832.65 | 4845.95 | 4846.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 13:15:00 | 4826.20 | 4842.00 | 4844.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 15:15:00 | 4842.00 | 4839.16 | 4842.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 4844.30 | 4840.19 | 4842.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 4844.30 | 4840.19 | 4842.66 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 4874.90 | 4842.28 | 4839.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 4890.00 | 4865.17 | 4852.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 10:15:00 | 5139.25 | 5145.27 | 5087.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 11:15:00 | 5119.35 | 5131.30 | 5110.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 5119.35 | 5131.30 | 5110.26 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 5447.50 | 5478.27 | 5479.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 11:15:00 | 5439.10 | 5470.44 | 5475.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 15:15:00 | 5392.55 | 5379.15 | 5409.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 5404.35 | 5384.19 | 5408.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 5404.35 | 5384.19 | 5408.62 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 5466.30 | 5422.80 | 5418.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 5506.00 | 5439.44 | 5426.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 5548.35 | 5567.80 | 5570.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 5489.70 | 5545.94 | 5558.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 11:15:00 | 5492.10 | 5478.61 | 5505.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 12:15:00 | 5503.80 | 5483.64 | 5505.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 5503.80 | 5483.64 | 5505.09 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 5528.25 | 5515.31 | 5514.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 14:15:00 | 5545.95 | 5527.62 | 5520.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 5500.00 | 5517.56 | 5518.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 5446.80 | 5503.41 | 5511.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 5434.10 | 5420.38 | 5451.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 15:15:00 | 5444.50 | 5425.20 | 5451.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 5444.50 | 5425.20 | 5451.20 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 13:15:00 | 5493.25 | 5469.19 | 5466.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 5511.55 | 5477.67 | 5470.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 10:15:00 | 5520.95 | 5537.51 | 5516.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 11:15:00 | 5509.15 | 5531.84 | 5515.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 5509.15 | 5531.84 | 5515.86 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 5401.15 | 5510.60 | 5522.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 5394.00 | 5487.28 | 5510.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 5522.00 | 5500.75 | 5499.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 5541.95 | 5512.77 | 5505.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 5694.00 | 5748.07 | 5750.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 5655.85 | 5729.63 | 5741.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 5786.00 | 5734.02 | 5731.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 5817.00 | 5750.62 | 5739.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 5772.80 | 5774.44 | 5756.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 10:15:00 | 5746.65 | 5768.88 | 5755.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 5746.65 | 5768.88 | 5755.16 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 6275.00 | 6307.37 | 6310.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 6248.05 | 6295.51 | 6304.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 11:15:00 | 6184.70 | 6179.38 | 6201.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 14:15:00 | 6234.90 | 6188.90 | 6200.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 6234.90 | 6188.90 | 6200.26 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 10:15:00 | 6295.00 | 6222.09 | 6213.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 11:15:00 | 6320.35 | 6241.75 | 6223.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 6684.00 | 6723.05 | 6660.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 6685.95 | 6715.63 | 6663.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 6685.95 | 6715.63 | 6663.20 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 09:15:00 | 6550.00 | 6635.35 | 6641.21 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 13:15:00 | 6658.20 | 6619.24 | 6618.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 6708.20 | 6646.72 | 6632.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 15:15:00 | 6759.65 | 6765.03 | 6728.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 6739.25 | 6759.87 | 6729.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 6739.25 | 6759.87 | 6729.32 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 6690.45 | 6740.64 | 6744.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 6660.40 | 6724.59 | 6736.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 6653.00 | 6643.76 | 6675.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 15:15:00 | 6064.05 | 6046.71 | 6065.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 6064.05 | 6046.71 | 6065.80 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 11:15:00 | 6115.20 | 6072.73 | 6072.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 6139.00 | 6085.98 | 6078.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 10:15:00 | 6147.20 | 6148.30 | 6117.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 11:15:00 | 6151.20 | 6148.88 | 6120.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 6151.20 | 6148.88 | 6120.47 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 6004.40 | 6105.63 | 6113.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 13:15:00 | 5988.20 | 6065.96 | 6092.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 6105.55 | 6040.67 | 6036.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 14:15:00 | 6138.95 | 6060.32 | 6045.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 6237.95 | 6273.06 | 6275.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 6225.00 | 6263.45 | 6271.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 10:15:00 | 6357.70 | 6284.79 | 6279.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 6417.35 | 6311.30 | 6292.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 6330.50 | 6333.54 | 6308.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 6362.50 | 6414.20 | 6403.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 6362.50 | 6414.20 | 6403.09 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 11:15:00 | 6366.65 | 6393.08 | 6394.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 6331.85 | 6372.89 | 6383.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 13:15:00 | 6450.90 | 6396.68 | 6391.77 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 6306.50 | 6383.77 | 6392.62 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 6493.20 | 6375.33 | 6370.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 12:15:00 | 6509.25 | 6402.12 | 6382.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 6416.80 | 6432.41 | 6433.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 6375.00 | 6415.24 | 6425.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 6316.70 | 6296.35 | 6335.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 14:15:00 | 6343.95 | 6305.87 | 6336.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 6343.95 | 6305.87 | 6336.59 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 6252.85 | 6224.92 | 6221.53 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 12:15:00 | 6204.25 | 6220.20 | 6220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 14:15:00 | 6195.20 | 6212.95 | 6216.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 6248.10 | 6221.48 | 6219.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 11:15:00 | 6275.80 | 6232.34 | 6224.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 5814.85 | 6189.44 | 6234.99 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 6037.60 | 6011.74 | 6008.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 14:15:00 | 6055.10 | 6028.63 | 6017.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 5893.40 | 5994.47 | 6005.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 5886.10 | 5972.80 | 5994.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 5833.75 | 5831.98 | 5866.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 10:15:00 | 5864.45 | 5838.47 | 5866.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 5864.45 | 5838.47 | 5866.11 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 5865.20 | 5842.65 | 5841.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 5900.00 | 5863.96 | 5855.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 5958.15 | 5959.37 | 5934.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 5932.15 | 5952.43 | 5935.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 5932.15 | 5952.43 | 5935.76 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 5867.45 | 5922.62 | 5926.48 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 5952.05 | 5919.63 | 5916.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 5975.10 | 5938.01 | 5925.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 5880.85 | 5918.64 | 5919.60 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 5970.00 | 5924.09 | 5920.71 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 5898.35 | 5922.74 | 5924.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 5888.65 | 5915.92 | 5921.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 5871.55 | 5846.13 | 5842.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 5931.55 | 5869.10 | 5855.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5763.70 | 5837.44 | 5842.58 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 5874.95 | 5847.27 | 5845.75 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 5826.00 | 5843.02 | 5843.95 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 5885.00 | 5852.10 | 5847.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 5944.25 | 5874.28 | 5859.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 6090.05 | 6099.46 | 6061.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 6197.75 | 6196.68 | 6174.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 6197.75 | 6196.68 | 6174.15 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 6159.95 | 6179.61 | 6180.66 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 6209.00 | 6176.44 | 6176.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 6242.75 | 6196.13 | 6186.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 6187.30 | 6237.26 | 6238.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 6136.85 | 6201.26 | 6220.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 6183.75 | 6139.76 | 6139.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 13:15:00 | 6198.00 | 6151.40 | 6144.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 6185.90 | 6187.28 | 6167.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 6227.60 | 6209.64 | 6187.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 6227.60 | 6209.64 | 6187.25 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 6401.00 | 6439.59 | 6443.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 6386.00 | 6428.87 | 6438.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 6422.65 | 6410.93 | 6410.46 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 6402.75 | 6409.30 | 6409.76 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 6595.20 | 6444.73 | 6424.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 6602.65 | 6476.31 | 6440.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 6635.60 | 6652.84 | 6594.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 6630.00 | 6644.18 | 6608.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 6630.00 | 6644.18 | 6608.61 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 6654.05 | 6687.16 | 6691.36 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 6746.25 | 6703.03 | 6697.34 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 6641.35 | 6687.84 | 6693.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 6598.45 | 6669.96 | 6685.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 6593.00 | 6573.95 | 6609.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 15:15:00 | 6555.00 | 6517.99 | 6554.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 6555.00 | 6517.99 | 6554.15 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 6606.60 | 6573.37 | 6572.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 09:15:00 | 6742.50 | 6613.27 | 6591.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 6780.00 | 6806.63 | 6807.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 6772.65 | 6799.83 | 6803.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 6804.00 | 6794.89 | 6794.48 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 6772.90 | 6793.71 | 6794.22 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 11:15:00 | 6812.00 | 6793.66 | 6793.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 12:15:00 | 6892.00 | 6813.32 | 6802.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 6908.45 | 6913.36 | 6879.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 6886.85 | 6908.06 | 6880.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 6886.85 | 6908.06 | 6880.60 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 6862.10 | 6879.28 | 6879.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 6838.05 | 6871.03 | 6875.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 6931.90 | 6880.57 | 6874.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 6991.25 | 6909.82 | 6889.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 6940.40 | 6947.99 | 6917.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 6935.60 | 6945.51 | 6919.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 6935.60 | 6945.51 | 6919.19 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 6880.00 | 6905.58 | 6908.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 6850.40 | 6894.54 | 6903.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 6933.40 | 6894.38 | 6893.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 6996.90 | 6923.40 | 6912.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 7010.20 | 7024.16 | 6992.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 7014.65 | 7036.53 | 7025.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 7014.65 | 7036.53 | 7025.11 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 6985.20 | 7021.94 | 7022.99 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 7038.95 | 7025.35 | 7024.44 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 7010.10 | 7022.30 | 7023.14 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 7067.95 | 7031.43 | 7027.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 10:15:00 | 7112.50 | 7047.64 | 7034.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 7165.75 | 7177.00 | 7177.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 15:15:00 | 7135.00 | 7168.60 | 7173.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 6925.00 | 6881.26 | 6881.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 7019.05 | 6908.82 | 6893.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 7045.90 | 7066.25 | 7068.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 7013.60 | 7055.72 | 7063.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 10:15:00 | 6950.70 | 6944.75 | 6944.58 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 6897.00 | 6935.20 | 6940.25 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 6959.50 | 6932.01 | 6931.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 6995.55 | 6944.72 | 6937.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 6972.55 | 6974.20 | 6957.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 6970.90 | 6974.65 | 6962.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 6970.90 | 6974.65 | 6962.16 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 6946.70 | 6981.72 | 6983.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 6908.45 | 6956.86 | 6970.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 6992.75 | 6955.54 | 6953.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 7417.40 | 7058.04 | 7002.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 09:15:00 | 7364.65 | 7393.88 | 7306.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 12:15:00 | 7176.00 | 7333.95 | 7299.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 7176.00 | 7333.95 | 7299.57 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 7155.00 | 7270.76 | 7275.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 7140.85 | 7205.92 | 7239.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 12:15:00 | 6782.45 | 6777.32 | 6830.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 15:15:00 | 6745.95 | 6728.45 | 6760.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 6745.95 | 6728.45 | 6760.91 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 6888.20 | 6799.46 | 6789.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 6897.10 | 6818.99 | 6798.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 7058.60 | 7069.17 | 7016.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 7032.50 | 7061.84 | 7018.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 7032.50 | 7061.84 | 7018.09 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 6887.15 | 6985.34 | 6996.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 6867.80 | 6961.84 | 6984.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 7013.95 | 6908.90 | 6908.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 7053.00 | 6937.72 | 6921.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 7188.85 | 7197.25 | 7136.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 11:15:00 | 7237.90 | 7205.38 | 7145.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 7237.90 | 7205.38 | 7145.29 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 7189.80 | 7201.45 | 7202.36 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 7280.00 | 7217.41 | 7209.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 7310.30 | 7259.11 | 7238.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 7224.30 | 7255.01 | 7257.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 7215.40 | 7242.79 | 7248.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 7267.80 | 7249.91 | 7249.21 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 7228.50 | 7249.79 | 7249.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 7220.00 | 7243.83 | 7247.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 7257.15 | 7239.07 | 7238.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 13:15:00 | 7295.35 | 7250.33 | 7243.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 13:15:00 | 7244.65 | 7270.93 | 7273.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 14:15:00 | 7233.45 | 7249.66 | 7259.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 7268.20 | 7260.64 | 7260.48 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 7252.00 | 7258.91 | 7259.71 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 7294.85 | 7258.09 | 7258.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 7319.50 | 7270.37 | 7263.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 7321.75 | 7362.71 | 7365.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 7295.00 | 7349.17 | 7358.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 12:15:00 | 7397.85 | 7362.34 | 7360.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 14:15:00 | 7430.00 | 7377.65 | 7367.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 7308.25 | 7388.15 | 7392.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 7225.00 | 7355.52 | 7377.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 13:15:00 | 6730.15 | 6720.85 | 6799.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 6729.25 | 6720.20 | 6779.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 6729.25 | 6720.20 | 6779.54 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 6823.65 | 6781.64 | 6781.52 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 6786.04 | 6787.68 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 6980.80 | 6824.99 | 6805.24 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 6830.70 | 6875.38 | 6879.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 6816.73 | 6848.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 6730.00 | 6714.26 | 6751.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 6745.00 | 6722.02 | 6743.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 6745.00 | 6722.02 | 6743.62 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 6785.60 | 6757.63 | 6755.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 6832.00 | 6777.04 | 6764.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 6814.45 | 6814.83 | 6789.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 6828.25 | 6824.92 | 6801.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 6828.25 | 6824.92 | 6801.11 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 6776.15 | 6822.73 | 6826.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 6709.80 | 6783.05 | 6804.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 6780.00 | 6771.17 | 6788.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 15:15:00 | 6786.00 | 6774.14 | 6788.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 6786.00 | 6774.14 | 6788.60 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 6904.15 | 6816.48 | 6806.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 6954.25 | 6844.03 | 6819.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 6795.80 | 6852.91 | 6857.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 6770.00 | 6836.33 | 6849.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 6758.90 | 6748.25 | 6786.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 15:15:00 | 6792.95 | 6757.19 | 6787.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 6792.95 | 6757.19 | 6787.04 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 6355.95 | 6326.59 | 6325.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 6414.50 | 6360.32 | 6343.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 6407.05 | 6416.95 | 6389.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 14:15:00 | 6396.65 | 6408.68 | 6392.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 6396.65 | 6408.68 | 6392.50 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 6335.30 | 6377.73 | 6381.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 6303.10 | 6338.71 | 6358.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 6085.00 | 6084.04 | 6145.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 6131.85 | 6093.60 | 6144.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 6131.85 | 6093.60 | 6144.20 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 6192.10 | 6162.33 | 6158.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 6227.30 | 6188.20 | 6176.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 6265.35 | 6267.02 | 6241.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 13:15:00 | 6250.00 | 6262.70 | 6244.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 6250.00 | 6262.70 | 6244.02 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 6181.50 | 6230.66 | 6232.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 6149.75 | 6184.57 | 6206.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 6128.85 | 6110.71 | 6150.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 6134.10 | 6115.39 | 6149.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 6134.10 | 6115.39 | 6149.33 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 6160.70 | 6128.16 | 6127.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 6252.60 | 6165.89 | 6146.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 6367.05 | 6372.60 | 6312.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 6627.70 | 6623.91 | 6564.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 6627.70 | 6623.91 | 6564.73 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 6519.15 | 6587.69 | 6589.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 6487.45 | 6554.62 | 6573.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 6633.05 | 6554.34 | 6549.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 6719.30 | 6633.75 | 6601.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 13:15:00 | 6723.40 | 6723.53 | 6685.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 14:15:00 | 6670.30 | 6712.88 | 6684.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 6670.30 | 6712.88 | 6684.31 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 6648.10 | 6677.37 | 6678.44 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 6714.50 | 6685.62 | 6681.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 6772.20 | 6702.94 | 6690.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 7014.00 | 7105.23 | 7106.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 6951.00 | 7074.38 | 7092.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 6997.50 | 6996.63 | 7034.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 12:15:00 | 7029.00 | 7002.76 | 7030.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 7029.00 | 7002.76 | 7030.69 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 7045.00 | 6992.80 | 6989.70 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 15:15:00 | 6970.00 | 7001.42 | 7002.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 09:15:00 | 6915.00 | 6984.13 | 6994.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 6927.00 | 6876.31 | 6870.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6995.50 | 6908.58 | 6886.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 6929.50 | 6930.00 | 6903.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 15:15:00 | 6926.00 | 6925.84 | 6907.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 6926.00 | 6925.84 | 6907.96 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 6876.00 | 6908.77 | 6910.65 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 6940.00 | 6911.46 | 6911.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 7013.50 | 6931.86 | 6920.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 15:15:00 | 7000.00 | 7013.02 | 6984.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 7051.50 | 7020.72 | 6991.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 7051.50 | 7020.72 | 6991.03 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 6930.50 | 6980.27 | 6983.19 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 6979.50 | 6975.06 | 6974.61 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 6933.50 | 6968.50 | 6971.81 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 7029.00 | 6971.17 | 6968.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 7043.00 | 6985.54 | 6975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 6986.50 | 7043.39 | 7049.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 6956.00 | 6998.52 | 7023.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 6880.00 | 6870.30 | 6869.28 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 6841.00 | 6868.03 | 6868.72 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 6880.00 | 6869.28 | 6869.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 6930.00 | 6881.42 | 6874.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 6897.50 | 6901.77 | 6887.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 6916.50 | 6904.71 | 6889.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 6916.50 | 6904.71 | 6889.79 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 6899.50 | 6912.06 | 6912.73 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 6926.50 | 6914.95 | 6913.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 6932.00 | 6918.36 | 6915.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 6985.50 | 6996.54 | 6975.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 7037.50 | 7066.60 | 7039.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 7037.50 | 7066.60 | 7039.20 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 6990.00 | 7028.58 | 7030.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 6946.50 | 7012.17 | 7022.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 7052.00 | 7010.20 | 7004.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 7052.50 | 7022.07 | 7011.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 7000.00 | 7021.19 | 7022.70 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 7054.50 | 7024.85 | 7023.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 7057.00 | 7034.94 | 7028.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 7498.00 | 7548.02 | 7554.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 7481.00 | 7534.61 | 7547.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 7236.00 | 7235.01 | 7306.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 11:15:00 | 7294.50 | 7248.60 | 7277.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 7294.50 | 7248.60 | 7277.87 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 7366.00 | 7297.64 | 7295.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 7396.50 | 7325.79 | 7308.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 7304.00 | 7342.48 | 7344.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 7253.00 | 7291.63 | 7312.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 7329.00 | 7287.41 | 7287.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 7344.00 | 7303.06 | 7294.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 7371.50 | 7393.78 | 7358.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 15:15:00 | 7366.00 | 7388.22 | 7359.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 7366.00 | 7388.22 | 7359.14 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 7357.00 | 7387.57 | 7390.44 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 7407.50 | 7392.50 | 7391.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 7427.00 | 7402.84 | 7396.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 7389.00 | 7426.40 | 7429.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 7334.00 | 7399.30 | 7416.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 7169.50 | 7144.14 | 7197.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 7133.50 | 7149.11 | 7190.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 7133.50 | 7149.11 | 7190.82 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 7259.50 | 7159.16 | 7156.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 7680.00 | 7321.51 | 7247.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 7819.50 | 7836.24 | 7731.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 7922.00 | 7913.07 | 7879.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 7922.00 | 7913.07 | 7879.65 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 7854.00 | 7885.44 | 7887.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 7848.50 | 7868.04 | 7878.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 7739.50 | 7695.68 | 7691.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 7762.00 | 7717.20 | 7703.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 7866.00 | 7867.73 | 7816.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 7846.00 | 7860.23 | 7821.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 7846.00 | 7860.23 | 7821.68 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 7786.50 | 7806.47 | 7807.84 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 7838.00 | 7810.04 | 7808.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 7859.00 | 7826.80 | 7817.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 7827.00 | 7863.54 | 7864.43 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 7899.00 | 7847.54 | 7844.72 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 7834.00 | 7853.04 | 7853.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 7825.00 | 7843.07 | 7848.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 7702.50 | 7700.63 | 7734.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 15:15:00 | 7450.50 | 7436.44 | 7463.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 7450.50 | 7436.44 | 7463.50 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 7591.50 | 7473.80 | 7467.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 7740.00 | 7695.37 | 7661.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 7697.50 | 7701.58 | 7673.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 7682.00 | 7697.05 | 7678.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 7682.00 | 7697.05 | 7678.37 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 7665.00 | 7678.75 | 7679.20 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 7718.50 | 7685.26 | 7681.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 7859.50 | 7813.34 | 7770.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 7902.00 | 7906.84 | 7859.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 8071.00 | 8011.90 | 7958.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8071.00 | 8011.90 | 7958.31 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 7872.00 | 7957.06 | 7960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 7853.50 | 7911.95 | 7937.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 7883.50 | 7883.36 | 7915.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 7848.50 | 7860.71 | 7886.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 7848.50 | 7860.71 | 7886.64 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 7835.00 | 7787.03 | 7782.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 7870.50 | 7803.72 | 7790.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 7820.50 | 7825.12 | 7806.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 7824.00 | 7824.89 | 7808.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 7824.00 | 7824.89 | 7808.40 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 7783.00 | 7801.68 | 7802.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 7732.50 | 7787.85 | 7796.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 15:15:00 | 7520.00 | 7505.05 | 7566.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 7454.00 | 7431.52 | 7457.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 7454.00 | 7431.52 | 7457.17 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 7492.50 | 7468.16 | 7466.17 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 7413.50 | 7457.23 | 7461.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 7382.00 | 7427.40 | 7443.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 7408.00 | 7404.32 | 7427.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 7456.00 | 7417.88 | 7429.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 7456.00 | 7417.88 | 7429.88 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 7457.00 | 7438.76 | 7437.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 7471.00 | 7447.89 | 7442.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 7410.00 | 7441.24 | 7442.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 7385.00 | 7413.67 | 7426.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 7416.00 | 7414.13 | 7425.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 7405.00 | 7382.61 | 7401.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 7405.00 | 7382.61 | 7401.21 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 7103.00 | 7061.99 | 7061.17 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 7021.00 | 7061.99 | 7062.28 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 7065.00 | 7062.47 | 7062.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 7076.50 | 7065.27 | 7063.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 7045.00 | 7068.74 | 7071.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 7010.00 | 7056.99 | 7065.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 7022.00 | 6983.97 | 6980.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 7085.00 | 7004.17 | 6989.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 7164.50 | 7170.60 | 7137.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 7156.50 | 7167.78 | 7139.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7156.50 | 7167.78 | 7139.52 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 7080.00 | 7126.85 | 7130.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 7030.50 | 7094.73 | 7113.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 7011.00 | 7006.35 | 7045.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 13:15:00 | 7040.00 | 7017.47 | 7041.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 7040.00 | 7017.47 | 7041.29 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 7083.50 | 7054.51 | 7051.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 7087.00 | 7061.01 | 7054.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 7092.50 | 7096.71 | 7080.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 7124.50 | 7109.86 | 7091.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7124.50 | 7109.86 | 7091.04 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 7256.00 | 7313.24 | 7320.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7183.50 | 7270.39 | 7297.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 7308.00 | 7280.74 | 7278.76 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 7273.00 | 7279.89 | 7280.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 6909.00 | 7016.22 | 7105.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 6891.00 | 6868.01 | 6953.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 6820.00 | 6793.39 | 6818.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 6820.00 | 6793.39 | 6818.37 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 6866.50 | 6836.73 | 6833.68 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 6807.00 | 6833.63 | 6834.67 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 6962.00 | 6843.44 | 6835.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 6989.00 | 6941.22 | 6903.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 6876.00 | 6902.22 | 6903.09 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 6928.00 | 6906.70 | 6904.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7034.50 | 6935.11 | 6918.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7072.00 | 7088.44 | 7049.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 11:15:00 | 7065.00 | 7079.60 | 7052.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 7065.00 | 7079.60 | 7052.25 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 7568.00 | 7595.58 | 7598.56 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 7622.50 | 7600.30 | 7599.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 7653.50 | 7612.20 | 7605.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 7720.50 | 7766.76 | 7767.49 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 7805.50 | 7768.37 | 7767.34 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 7556.50 | 7727.86 | 7749.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 7497.50 | 7656.61 | 7711.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 7792.50 | 7707.14 | 7698.09 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 7626.50 | 7702.97 | 7703.47 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 7776.00 | 7712.05 | 7704.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 7827.50 | 7747.77 | 7722.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 7698.00 | 7744.91 | 7744.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 7667.00 | 7723.03 | 7734.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 7598.00 | 7585.83 | 7634.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 7548.50 | 7560.16 | 7601.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 7548.50 | 7560.16 | 7601.15 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 7431.50 | 7304.06 | 7303.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7541.00 | 7396.69 | 7351.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 7533.00 | 7536.52 | 7461.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 7600.50 | 7549.32 | 7473.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 7618.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 7602.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | SL hit (close<ema400) qty=1.00 sl=7498.99 alert=retest1 |

### Cycle 226 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 7394.00 | 7464.66 | 7473.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 7306.50 | 7424.00 | 7449.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 7346.50 | 7295.50 | 7345.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 7325.50 | 7301.50 | 7344.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 7330.00 | 7301.50 | 7344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 7274.50 | 7295.70 | 7334.02 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 7381.00 | 7335.74 | 7332.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 7545.50 | 7480.84 | 7434.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 7506.50 | 7507.91 | 7464.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7518.00 | 7509.37 | 7475.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 7777.00 | 7763.78 | 7762.57 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 7753.50 | 7761.72 | 7761.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 7722.00 | 7752.54 | 7757.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 7762.00 | 7663.53 | 7686.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 7760.00 | 7682.82 | 7692.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 7776.00 | 7682.82 | 7692.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 7737.00 | 7704.33 | 7701.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 7774.50 | 7738.69 | 7723.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 7755.00 | 7759.20 | 7745.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 7765.50 | 7759.20 | 7745.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 7768.50 | 7761.06 | 7747.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 7760.00 | 7761.06 | 7747.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-27 11:00:00 | 7600.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-03-27 11:30:00 | 7618.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 7602.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 7518.00 | 2026-04-28 15:15:00 | 7740.00 | STOP_HIT | 1.00 | 2.95% |
