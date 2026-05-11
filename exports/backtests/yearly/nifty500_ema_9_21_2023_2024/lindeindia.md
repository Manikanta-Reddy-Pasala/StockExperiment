# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 7765.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 142 |
| ALERT2 | 138 |
| ALERT2_SKIP | 93 |
| ALERT3 | 285 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 161 |
| PARTIAL | 32 |
| TARGET_HIT | 4 |
| STOP_HIT | 166 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 202 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 87 / 115
- **Target hits / Stop hits / Partials:** 4 / 166 / 32
- **Avg / median % per leg:** 0.59% / -0.59%
- **Sum % (uncompounded):** 118.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 12 | 16.9% | 4 | 65 | 2 | -0.54% | -38.3% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.44% | 10.1% |
| BUY @ 3rd Alert (retest2) | 64 | 8 | 12.5% | 4 | 60 | 0 | -0.76% | -48.4% |
| SELL (all) | 131 | 75 | 57.3% | 0 | 101 | 30 | 1.20% | 156.7% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -1.73% | -8.6% |
| SELL @ 3rd Alert (retest2) | 126 | 73 | 57.9% | 0 | 97 | 29 | 1.31% | 165.3% |
| retest1 (combined) | 12 | 6 | 50.0% | 0 | 9 | 3 | 0.12% | 1.4% |
| retest2 (combined) | 190 | 81 | 42.6% | 4 | 157 | 29 | 0.62% | 116.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 3987.00 | 3951.60 | 3950.48 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 11:15:00 | 3939.15 | 3950.89 | 3951.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 14:15:00 | 3930.25 | 3942.10 | 3947.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 3919.90 | 3907.86 | 3920.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 3919.90 | 3907.86 | 3920.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 3919.90 | 3907.86 | 3920.62 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 3998.60 | 3928.84 | 3922.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 12:15:00 | 4008.00 | 3956.05 | 3936.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 3961.40 | 3975.14 | 3953.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 10:15:00 | 3961.60 | 3972.43 | 3954.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 3961.60 | 3972.43 | 3954.55 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 3967.80 | 3981.27 | 3981.97 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 12:15:00 | 3990.00 | 3983.09 | 3982.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 13:15:00 | 3999.80 | 3986.43 | 3984.14 | Break + close above crossover candle high |

### Cycle 6 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 3960.75 | 3981.30 | 3982.02 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 11:15:00 | 3995.00 | 3982.26 | 3981.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 4004.95 | 3993.01 | 3987.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 15:15:00 | 3978.00 | 3996.29 | 3992.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 15:15:00 | 3978.00 | 3996.29 | 3992.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 3978.00 | 3996.29 | 3992.63 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 3987.95 | 3993.17 | 3993.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 12:15:00 | 3979.00 | 3990.33 | 3992.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 3997.75 | 3986.95 | 3989.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 3997.75 | 3986.95 | 3989.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 3997.75 | 3986.95 | 3989.59 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 11:15:00 | 3992.00 | 3978.58 | 3977.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 12:15:00 | 3999.75 | 3982.81 | 3979.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 11:15:00 | 4517.70 | 4522.95 | 4423.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 15:15:00 | 4438.00 | 4490.76 | 4438.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 4438.00 | 4490.76 | 4438.98 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 11:15:00 | 4385.90 | 4446.58 | 4448.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 14:15:00 | 4368.60 | 4413.69 | 4431.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 4543.80 | 4398.91 | 4404.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 4543.80 | 4398.91 | 4404.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 4543.80 | 4398.91 | 4404.04 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 10:15:00 | 4592.15 | 4437.56 | 4421.14 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 4400.80 | 4482.48 | 4483.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 15:15:00 | 4357.50 | 4410.45 | 4443.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 4403.90 | 4397.54 | 4417.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 4403.90 | 4397.54 | 4417.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 4403.90 | 4397.54 | 4417.58 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 09:15:00 | 4431.55 | 4357.11 | 4349.75 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 4342.25 | 4371.10 | 4374.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 4301.15 | 4344.05 | 4358.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 4375.25 | 4330.09 | 4340.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 4375.25 | 4330.09 | 4340.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 4375.25 | 4330.09 | 4340.86 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 4398.90 | 4354.63 | 4350.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 4401.25 | 4363.95 | 4355.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 15:15:00 | 4436.45 | 4439.27 | 4411.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 4427.65 | 4436.95 | 4413.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 4427.65 | 4436.95 | 4413.41 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 09:15:00 | 4315.05 | 4392.37 | 4402.56 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 4441.15 | 4391.56 | 4388.01 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 14:15:00 | 4369.50 | 4402.02 | 4402.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 15:15:00 | 4351.00 | 4391.82 | 4398.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 10:15:00 | 4404.55 | 4389.19 | 4395.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 10:15:00 | 4404.55 | 4389.19 | 4395.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 4404.55 | 4389.19 | 4395.42 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 12:15:00 | 4515.00 | 4418.44 | 4407.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 14:15:00 | 4598.50 | 4471.66 | 4434.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 14:15:00 | 4587.75 | 4590.93 | 4550.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 15:15:00 | 4612.00 | 4662.88 | 4623.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 4612.00 | 4662.88 | 4623.32 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 14:15:00 | 5042.20 | 5069.64 | 5072.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 14:15:00 | 4949.40 | 5002.05 | 5033.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 11:15:00 | 4991.40 | 4982.17 | 5012.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 12:15:00 | 5031.80 | 4992.09 | 5014.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 5031.80 | 4992.09 | 5014.19 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 5029.10 | 4958.95 | 4949.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 10:15:00 | 5055.00 | 4997.73 | 4977.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 4999.85 | 5003.66 | 4987.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 15:15:00 | 4985.50 | 5000.03 | 4987.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 4985.50 | 5000.03 | 4987.13 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 15:15:00 | 6499.00 | 6548.49 | 6549.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 09:15:00 | 6460.20 | 6530.83 | 6540.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 6493.10 | 6455.91 | 6489.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 6493.10 | 6455.91 | 6489.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 6493.10 | 6455.91 | 6489.59 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 14:15:00 | 6543.95 | 6510.40 | 6507.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 6559.70 | 6523.55 | 6513.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 12:15:00 | 6492.60 | 6520.77 | 6515.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 12:15:00 | 6492.60 | 6520.77 | 6515.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 6492.60 | 6520.77 | 6515.49 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 14:15:00 | 6474.75 | 6508.23 | 6510.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 6290.70 | 6457.81 | 6486.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 6088.75 | 6081.44 | 6179.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 14:15:00 | 5980.15 | 5954.81 | 6004.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 5980.15 | 5954.81 | 6004.60 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 5960.00 | 5915.08 | 5914.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 09:15:00 | 6002.05 | 5932.48 | 5922.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 5973.85 | 6006.39 | 5982.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 5973.85 | 6006.39 | 5982.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 5973.85 | 6006.39 | 5982.54 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 13:15:00 | 5954.25 | 5974.12 | 5975.88 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 6015.35 | 5978.21 | 5976.49 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 5951.60 | 5974.19 | 5975.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 5934.00 | 5966.28 | 5971.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 5985.60 | 5965.66 | 5969.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 14:15:00 | 5985.60 | 5965.66 | 5969.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 5985.60 | 5965.66 | 5969.64 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 5991.30 | 5972.32 | 5971.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 5938.55 | 5965.57 | 5968.80 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 6005.05 | 5976.42 | 5973.21 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 5936.65 | 5976.87 | 5978.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 12:15:00 | 5929.10 | 5947.76 | 5960.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 5921.70 | 5884.36 | 5907.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 5921.70 | 5884.36 | 5907.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 5921.70 | 5884.36 | 5907.41 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 13:15:00 | 6498.15 | 6030.52 | 5968.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 14:15:00 | 6671.75 | 6158.77 | 6032.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 14:15:00 | 6419.45 | 6432.29 | 6343.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 14:15:00 | 6397.45 | 6434.12 | 6390.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 6397.45 | 6434.12 | 6390.45 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 6370.00 | 6384.73 | 6384.88 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 14:15:00 | 6535.95 | 6390.43 | 6383.65 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 6229.75 | 6395.84 | 6403.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 6216.75 | 6336.11 | 6373.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 15:15:00 | 5951.85 | 5943.72 | 6043.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 15:15:00 | 5951.85 | 5943.72 | 6043.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 5951.85 | 5943.72 | 6043.02 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 6209.75 | 6016.15 | 5999.10 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 6140.20 | 6160.40 | 6161.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 13:15:00 | 6102.35 | 6148.79 | 6156.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 6115.00 | 6074.35 | 6098.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 6115.00 | 6074.35 | 6098.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 6115.00 | 6074.35 | 6098.79 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 6042.65 | 6032.32 | 6032.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 11:15:00 | 6088.10 | 6047.04 | 6039.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 6063.55 | 6070.96 | 6055.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 6063.55 | 6070.96 | 6055.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 6063.55 | 6070.96 | 6055.97 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 15:15:00 | 6030.00 | 6049.74 | 6051.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 09:15:00 | 6024.60 | 6044.72 | 6049.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 5974.85 | 5951.21 | 5967.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 5974.85 | 5951.21 | 5967.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 5974.85 | 5951.21 | 5967.81 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 5946.00 | 5821.60 | 5815.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 12:15:00 | 5974.45 | 5912.82 | 5880.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 5944.10 | 5948.48 | 5910.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 12:15:00 | 5908.00 | 5939.48 | 5915.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 5908.00 | 5939.48 | 5915.77 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 5869.00 | 5905.23 | 5907.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 15:15:00 | 5865.00 | 5890.43 | 5899.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 5924.90 | 5897.32 | 5901.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 5924.90 | 5897.32 | 5901.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 5924.90 | 5897.32 | 5901.88 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 12:15:00 | 5718.55 | 5692.51 | 5691.48 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 5670.00 | 5690.57 | 5690.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 09:15:00 | 5630.00 | 5675.61 | 5683.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 12:15:00 | 5680.70 | 5668.94 | 5677.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 12:15:00 | 5680.70 | 5668.94 | 5677.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 5680.70 | 5668.94 | 5677.96 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 14:15:00 | 5748.65 | 5650.20 | 5643.79 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 11:15:00 | 5655.05 | 5673.03 | 5673.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 12:15:00 | 5621.00 | 5662.63 | 5668.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-28 10:15:00 | 5657.85 | 5627.95 | 5644.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 10:15:00 | 5657.85 | 5627.95 | 5644.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 5657.85 | 5627.95 | 5644.70 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 15:15:00 | 5685.00 | 5650.88 | 5650.36 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 5628.05 | 5648.84 | 5650.49 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 5682.95 | 5654.53 | 5652.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 12:15:00 | 5702.90 | 5667.16 | 5658.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 15:15:00 | 5645.75 | 5669.37 | 5662.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 15:15:00 | 5645.75 | 5669.37 | 5662.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 5645.75 | 5669.37 | 5662.31 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 5606.00 | 5656.69 | 5657.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 5601.85 | 5645.73 | 5652.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 5556.20 | 5542.67 | 5575.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 5556.20 | 5542.67 | 5575.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 5556.20 | 5542.67 | 5575.82 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 5621.05 | 5593.75 | 5590.49 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 5580.00 | 5590.39 | 5590.79 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 11:15:00 | 5600.00 | 5587.68 | 5586.53 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 5569.60 | 5584.07 | 5584.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 13:15:00 | 5564.85 | 5580.22 | 5583.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 5552.00 | 5547.36 | 5558.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 5552.00 | 5547.36 | 5558.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 5552.00 | 5547.36 | 5558.57 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 14:15:00 | 5555.00 | 5549.62 | 5549.14 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 5536.25 | 5548.10 | 5548.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 5472.00 | 5530.75 | 5540.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 5490.75 | 5473.46 | 5500.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 13:15:00 | 5511.90 | 5482.78 | 5499.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 5511.90 | 5482.78 | 5499.87 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 5710.00 | 5535.31 | 5514.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 5892.00 | 5634.73 | 5565.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 5780.00 | 5787.82 | 5703.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 5738.30 | 5778.28 | 5713.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 5738.30 | 5778.28 | 5713.47 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 5630.00 | 5694.20 | 5694.27 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 13:15:00 | 5720.25 | 5695.38 | 5693.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 11:15:00 | 5750.00 | 5722.45 | 5709.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 14:15:00 | 5717.85 | 5727.06 | 5715.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 14:15:00 | 5717.85 | 5727.06 | 5715.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 5717.85 | 5727.06 | 5715.04 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 09:15:00 | 5682.30 | 5709.32 | 5711.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 5641.50 | 5679.34 | 5694.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 5555.00 | 5542.81 | 5583.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 5555.00 | 5542.81 | 5583.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 5555.00 | 5542.81 | 5583.70 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 10:15:00 | 5689.55 | 5598.75 | 5590.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 5725.00 | 5666.08 | 5636.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 09:15:00 | 5647.00 | 5667.69 | 5642.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 10:15:00 | 5650.80 | 5664.31 | 5643.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 5650.80 | 5664.31 | 5643.29 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 5588.70 | 5646.61 | 5650.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 5569.45 | 5616.29 | 5634.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 5612.90 | 5597.52 | 5617.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 14:15:00 | 5612.90 | 5597.52 | 5617.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 5612.90 | 5597.52 | 5617.81 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 5577.85 | 5560.39 | 5559.78 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 14:15:00 | 5550.00 | 5558.31 | 5558.89 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 5565.95 | 5560.27 | 5559.71 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 5533.75 | 5554.96 | 5557.35 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 13:15:00 | 5600.00 | 5562.16 | 5559.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 14:15:00 | 5621.00 | 5573.93 | 5565.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 5754.85 | 5761.17 | 5709.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 12:15:00 | 5736.00 | 5753.10 | 5718.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 5736.00 | 5753.10 | 5718.74 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 5676.60 | 5702.76 | 5704.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 5655.00 | 5689.28 | 5698.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 09:15:00 | 5568.90 | 5565.14 | 5596.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 5568.90 | 5565.14 | 5596.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 5568.90 | 5565.14 | 5596.57 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 5535.80 | 5510.04 | 5507.49 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 5474.00 | 5504.89 | 5508.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 5425.65 | 5485.38 | 5498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 14:15:00 | 5459.00 | 5451.90 | 5470.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 15:15:00 | 5460.00 | 5453.52 | 5469.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 5460.00 | 5453.52 | 5469.87 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 10:15:00 | 5600.00 | 5487.64 | 5482.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 5726.95 | 5610.79 | 5572.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 13:15:00 | 6194.65 | 6253.41 | 6104.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 6065.55 | 6215.84 | 6100.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 6065.55 | 6215.84 | 6100.93 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 13:15:00 | 6494.00 | 6622.50 | 6624.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 14:15:00 | 6463.05 | 6590.61 | 6609.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 13:15:00 | 6547.60 | 6511.96 | 6554.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 13:15:00 | 6547.60 | 6511.96 | 6554.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 13:15:00 | 6547.60 | 6511.96 | 6554.13 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 6613.25 | 6569.71 | 6567.19 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 13:15:00 | 6540.90 | 6563.95 | 6564.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 14:15:00 | 6525.55 | 6556.27 | 6561.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 6396.70 | 6394.11 | 6431.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 10:15:00 | 6558.00 | 6426.89 | 6443.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 6558.00 | 6426.89 | 6443.40 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 6480.45 | 6452.40 | 6449.86 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 15:15:00 | 6420.00 | 6444.43 | 6446.92 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 12:15:00 | 6511.80 | 6454.47 | 6448.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 6880.65 | 6551.72 | 6494.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 7070.80 | 7114.29 | 6959.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 10:15:00 | 7006.00 | 7063.62 | 6971.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 7006.00 | 7063.62 | 6971.93 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 6924.10 | 6944.83 | 6946.93 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 6996.85 | 6955.23 | 6951.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 7065.00 | 6977.19 | 6961.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-08 13:15:00 | 6988.00 | 6989.90 | 6970.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 14:15:00 | 6921.00 | 6976.12 | 6966.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 6921.00 | 6976.12 | 6966.36 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 09:15:00 | 6923.90 | 6959.89 | 6960.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 10:15:00 | 6872.00 | 6942.32 | 6952.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 6889.85 | 6889.22 | 6917.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 6889.85 | 6889.22 | 6917.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 6889.85 | 6889.22 | 6917.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 7020.00 | 6919.48 | 6920.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 6927.25 | 6921.03 | 6920.68 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 6915.00 | 6919.83 | 6920.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 11:15:00 | 6902.05 | 6916.27 | 6918.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 6883.95 | 6820.20 | 6844.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 6883.95 | 6820.20 | 6844.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 6883.95 | 6820.20 | 6844.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 6883.95 | 6820.20 | 6844.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 6900.00 | 6836.16 | 6849.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:30:00 | 6900.00 | 6836.16 | 6849.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 6908.80 | 6866.91 | 6861.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 7100.00 | 6925.80 | 6890.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 7056.55 | 7097.28 | 7016.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 7056.55 | 7097.28 | 7016.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 7056.55 | 7097.28 | 7016.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:45:00 | 7150.00 | 7105.83 | 7034.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-24 09:15:00 | 7865.00 | 7508.34 | 7367.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 14:15:00 | 8270.35 | 8340.65 | 8343.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 8381.70 | 8345.28 | 8343.00 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 8271.35 | 8341.37 | 8347.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 8068.90 | 8286.87 | 8321.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 7779.90 | 7682.70 | 7813.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 7779.90 | 7682.70 | 7813.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 7779.90 | 7682.70 | 7813.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:45:00 | 7771.60 | 7682.70 | 7813.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 7790.10 | 7746.66 | 7806.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 7793.60 | 7746.66 | 7806.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 7795.00 | 7756.33 | 7805.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 7794.65 | 7756.33 | 7805.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 7825.00 | 7770.06 | 7807.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 7870.30 | 7770.06 | 7807.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 7765.00 | 7769.05 | 7803.51 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 7919.20 | 7821.75 | 7818.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 7931.00 | 7843.60 | 7828.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 8984.70 | 8985.40 | 8713.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:15:00 | 9238.25 | 8996.09 | 8742.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:15:00 | 9700.16 | 9456.52 | 9207.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 9620.00 | 9641.25 | 9450.63 | SL hit (close<ema200) qty=0.50 sl=9641.25 alert=retest1 |

### Cycle 88 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 9348.75 | 9386.24 | 9390.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 12:15:00 | 9145.00 | 9293.10 | 9343.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 15:15:00 | 9135.00 | 9133.87 | 9205.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 09:15:00 | 9069.95 | 9133.87 | 9205.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 8924.95 | 9092.09 | 9180.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 13:30:00 | 8807.70 | 8951.99 | 9079.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 15:15:00 | 8367.32 | 8497.23 | 8718.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 8495.30 | 8417.03 | 8618.42 | SL hit (close>ema200) qty=0.50 sl=8417.03 alert=retest2 |

### Cycle 89 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 9063.60 | 8632.43 | 8623.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 12:15:00 | 9160.00 | 8737.94 | 8671.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 09:15:00 | 8784.00 | 8890.76 | 8780.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 8784.00 | 8890.76 | 8780.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 8784.00 | 8890.76 | 8780.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 8784.00 | 8890.76 | 8780.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 8826.00 | 8877.81 | 8784.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:30:00 | 8809.00 | 8877.81 | 8784.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 8787.75 | 8859.80 | 8784.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:45:00 | 8789.75 | 8859.80 | 8784.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 8779.55 | 8843.75 | 8784.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:30:00 | 8758.75 | 8843.75 | 8784.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 8801.60 | 8835.32 | 8785.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:30:00 | 8773.00 | 8835.32 | 8785.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 8759.55 | 8820.16 | 8783.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 8759.55 | 8820.16 | 8783.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 8675.00 | 8791.13 | 8773.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 8968.75 | 8791.13 | 8773.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:00:00 | 8774.00 | 8801.14 | 8782.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 8771.00 | 8793.31 | 8780.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 12:15:00 | 8774.50 | 8793.31 | 8780.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 8855.80 | 8804.25 | 8787.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:30:00 | 8842.65 | 8804.25 | 8787.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 8850.10 | 8836.48 | 8808.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 8779.15 | 8836.48 | 8808.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 8290.45 | 8727.27 | 8761.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 8290.45 | 8727.27 | 8761.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 8120.00 | 8605.82 | 8703.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 8455.65 | 8209.57 | 8339.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 8455.65 | 8209.57 | 8339.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 8455.65 | 8209.57 | 8339.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 8455.65 | 8209.57 | 8339.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 8763.75 | 8320.40 | 8377.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 8767.30 | 8320.40 | 8377.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 8698.00 | 8450.26 | 8430.15 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 13:15:00 | 8493.65 | 8520.56 | 8521.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-10 14:15:00 | 8452.20 | 8506.89 | 8515.31 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 8880.00 | 8568.43 | 8541.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 12:15:00 | 9041.20 | 8713.69 | 8618.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 9169.25 | 9195.94 | 8986.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 9169.25 | 9195.94 | 8986.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 9146.20 | 9269.70 | 9166.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 9146.20 | 9269.70 | 9166.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 9236.00 | 9262.96 | 9172.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:30:00 | 9259.55 | 9261.14 | 9180.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 9530.05 | 9256.11 | 9203.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 9275.00 | 9398.25 | 9320.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 9055.00 | 9309.88 | 9292.96 | SL hit (close<static) qty=1.00 sl=9135.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 8999.65 | 9247.83 | 9266.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 8932.00 | 9058.68 | 9154.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 9070.60 | 9061.07 | 9146.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 9070.60 | 9061.07 | 9146.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 9070.60 | 9061.07 | 9146.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 9007.60 | 9061.07 | 9146.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 8402.95 | 8344.66 | 8492.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 8360.00 | 8352.53 | 8482.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 8335.85 | 8356.02 | 8462.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 8286.25 | 8232.55 | 8261.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 8423.10 | 8301.25 | 8285.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 8423.10 | 8301.25 | 8285.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 11:15:00 | 8522.95 | 8443.75 | 8384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 12:15:00 | 8979.35 | 8991.53 | 8822.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:00:00 | 8979.35 | 8991.53 | 8822.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 9240.95 | 9054.91 | 8907.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 9263.80 | 9054.91 | 8907.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 9264.65 | 9131.56 | 8981.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:15:00 | 9253.80 | 9152.78 | 9005.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 8879.95 | 9073.52 | 9060.74 | SL hit (close<static) qty=1.00 sl=8894.80 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 8766.70 | 9012.16 | 9034.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 8490.00 | 8625.75 | 8687.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 8167.55 | 8151.67 | 8292.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:00:00 | 8167.55 | 8151.67 | 8292.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 7970.00 | 8008.62 | 8097.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 8053.40 | 8008.62 | 8097.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 7903.90 | 7935.01 | 8016.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:30:00 | 7919.15 | 7935.01 | 8016.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 8090.95 | 7966.20 | 8023.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 8090.95 | 7966.20 | 8023.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 8335.05 | 8039.97 | 8051.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 8335.05 | 8039.97 | 8051.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 8171.20 | 8066.21 | 8062.76 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 13:15:00 | 7917.25 | 8041.88 | 8057.05 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 8440.60 | 8068.93 | 8023.25 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 8061.00 | 8118.63 | 8120.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 7856.00 | 8056.97 | 8091.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 7850.00 | 7825.27 | 7923.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 7850.00 | 7825.27 | 7923.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 7850.00 | 7825.27 | 7923.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:30:00 | 7798.30 | 7830.78 | 7901.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 7798.15 | 7819.82 | 7883.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 7767.85 | 7801.83 | 7837.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 7408.39 | 7638.83 | 7703.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 7408.24 | 7638.83 | 7703.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 7617.75 | 7607.39 | 7669.86 | SL hit (close>ema200) qty=0.50 sl=7607.39 alert=retest2 |

### Cycle 101 — BUY (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 14:15:00 | 7425.00 | 7331.98 | 7321.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 7480.00 | 7361.59 | 7336.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 13:15:00 | 7465.00 | 7469.38 | 7409.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 13:45:00 | 7472.35 | 7469.38 | 7409.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 7421.00 | 7453.36 | 7412.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 7379.40 | 7453.36 | 7412.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 7335.05 | 7429.70 | 7405.16 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 7280.25 | 7377.50 | 7384.31 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 7395.65 | 7348.09 | 7344.36 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 7299.00 | 7334.74 | 7338.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 7265.00 | 7313.63 | 7327.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 13:15:00 | 7285.05 | 7257.83 | 7291.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 13:15:00 | 7285.05 | 7257.83 | 7291.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 7285.05 | 7257.83 | 7291.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:00:00 | 7285.05 | 7257.83 | 7291.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 7298.00 | 7265.87 | 7292.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 7269.95 | 7276.26 | 7290.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:30:00 | 7269.95 | 7277.00 | 7289.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:15:00 | 7272.00 | 7276.80 | 7288.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 7260.00 | 7278.66 | 7287.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 7260.00 | 7274.93 | 7285.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 7259.95 | 7274.93 | 7285.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 7260.15 | 7271.97 | 7282.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 14:15:00 | 7197.30 | 7233.45 | 7258.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 15:15:00 | 7191.00 | 7226.69 | 7253.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 12:15:00 | 7341.15 | 7242.17 | 7248.88 | SL hit (close>static) qty=1.00 sl=7315.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 7339.60 | 7261.65 | 7257.13 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 7230.00 | 7250.26 | 7252.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 12:15:00 | 7201.60 | 7236.78 | 7245.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 7227.95 | 7220.81 | 7233.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 7227.95 | 7220.81 | 7233.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 7227.95 | 7220.81 | 7233.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 7207.45 | 7220.81 | 7233.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 7218.60 | 7220.37 | 7232.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 7193.20 | 7217.84 | 7226.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:00:00 | 7203.90 | 7215.05 | 7224.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 7526.80 | 7276.87 | 7250.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 7526.80 | 7276.87 | 7250.15 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 7300.05 | 7348.93 | 7350.38 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 7377.00 | 7350.77 | 7350.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 7430.00 | 7372.09 | 7360.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 7740.90 | 7742.76 | 7611.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 12:30:00 | 7747.05 | 7742.76 | 7611.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 7635.00 | 7716.48 | 7631.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 7603.00 | 7716.48 | 7631.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 7595.30 | 7692.25 | 7628.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 7597.50 | 7692.25 | 7628.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 7620.90 | 7677.98 | 7627.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 7800.75 | 7719.95 | 7662.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 11:15:00 | 8580.83 | 8240.93 | 8206.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 8436.85 | 8537.53 | 8549.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 8414.90 | 8473.07 | 8496.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 8189.95 | 8148.75 | 8245.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 7958.90 | 8099.01 | 8214.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 8171.05 | 7999.11 | 8054.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 8171.05 | 7999.11 | 8054.59 | SL hit (close>ema400) qty=1.00 sl=8054.59 alert=retest1 |

### Cycle 111 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 8153.10 | 8095.96 | 8088.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 8223.90 | 8144.15 | 8114.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 8238.70 | 8243.53 | 8187.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 15:00:00 | 8322.30 | 8259.28 | 8199.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 8241.00 | 8265.34 | 8213.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 8204.90 | 8249.22 | 8214.87 | SL hit (close<ema400) qty=1.00 sl=8214.87 alert=retest1 |

### Cycle 112 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 8183.35 | 8201.03 | 8202.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 8093.85 | 8179.59 | 8192.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 8301.60 | 8180.45 | 8185.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 8301.60 | 8180.45 | 8185.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 8301.60 | 8180.45 | 8185.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 8301.60 | 8180.45 | 8185.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 8353.45 | 8215.05 | 8201.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 14:15:00 | 8466.20 | 8330.69 | 8287.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 8310.00 | 8344.04 | 8301.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 8310.00 | 8344.04 | 8301.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 8310.00 | 8344.04 | 8301.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 8310.00 | 8344.04 | 8301.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 8300.00 | 8335.23 | 8301.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 8337.70 | 8335.23 | 8301.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 8230.00 | 8314.19 | 8295.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 8232.90 | 8314.19 | 8295.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 8244.55 | 8300.26 | 8290.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 8239.00 | 8300.26 | 8290.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 8194.55 | 8279.12 | 8281.77 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 8390.55 | 8301.40 | 8291.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 12:15:00 | 8425.50 | 8346.90 | 8319.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 15:15:00 | 8357.00 | 8363.66 | 8335.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:15:00 | 8384.95 | 8363.66 | 8335.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 8334.45 | 8357.82 | 8335.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 8299.00 | 8357.82 | 8335.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 8306.10 | 8347.47 | 8332.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 8301.25 | 8347.47 | 8332.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 8241.90 | 8326.36 | 8324.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:30:00 | 8241.45 | 8326.36 | 8324.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 8212.35 | 8303.56 | 8314.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 8178.10 | 8265.10 | 8294.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 14:15:00 | 7976.00 | 7970.31 | 8057.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 14:45:00 | 7990.15 | 7970.31 | 8057.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 8016.90 | 7976.72 | 8044.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 7938.50 | 7973.49 | 8031.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:45:00 | 7941.00 | 7967.68 | 8023.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 7930.00 | 7966.13 | 8017.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 7875.65 | 7967.93 | 8009.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 7800.20 | 7934.39 | 7990.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:45:00 | 7774.30 | 7895.88 | 7968.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:00:00 | 7788.30 | 7871.50 | 7944.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 7770.00 | 7845.30 | 7925.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 7541.57 | 7730.44 | 7848.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 7543.95 | 7730.44 | 7848.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 7533.50 | 7730.44 | 7848.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 7481.87 | 7730.44 | 7848.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 15:15:00 | 7570.00 | 7532.57 | 7617.79 | SL hit (close>ema200) qty=0.50 sl=7532.57 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 7653.10 | 7615.74 | 7611.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 7800.00 | 7658.94 | 7632.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 7680.05 | 7695.88 | 7660.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:30:00 | 7680.50 | 7695.88 | 7660.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 7759.95 | 7708.69 | 7669.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 7773.25 | 7720.06 | 7688.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 7831.75 | 7759.69 | 7712.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 15:00:00 | 7795.15 | 7883.68 | 7858.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 7648.85 | 7821.41 | 7833.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 7648.85 | 7821.41 | 7833.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 7545.15 | 7658.77 | 7739.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 6977.05 | 6854.70 | 7040.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 6977.05 | 6854.70 | 7040.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 6731.25 | 6814.18 | 6930.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:45:00 | 6635.00 | 6776.01 | 6902.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:30:00 | 6630.10 | 6749.97 | 6879.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 6626.00 | 6724.61 | 6855.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 6629.40 | 6660.65 | 6734.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 6740.05 | 6538.64 | 6550.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 6797.00 | 6538.64 | 6550.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 6733.00 | 6577.51 | 6567.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 6733.00 | 6577.51 | 6567.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 6796.00 | 6716.86 | 6662.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 6965.65 | 6973.87 | 6873.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:45:00 | 6953.10 | 6973.87 | 6873.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 6895.30 | 6942.28 | 6894.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 6895.30 | 6942.28 | 6894.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 6936.00 | 6941.02 | 6898.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 6942.70 | 6923.19 | 6899.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:00:00 | 6979.00 | 6929.52 | 6910.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 13:15:00 | 6902.50 | 6945.48 | 6950.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 6902.50 | 6945.48 | 6950.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 6868.80 | 6916.39 | 6934.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 6874.60 | 6856.79 | 6887.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 6874.60 | 6856.79 | 6887.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 6874.60 | 6856.79 | 6887.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:15:00 | 6923.60 | 6856.79 | 6887.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 6869.55 | 6859.34 | 6885.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:45:00 | 6909.25 | 6859.34 | 6885.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 6840.00 | 6842.71 | 6865.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 6827.15 | 6842.71 | 6865.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:45:00 | 6827.00 | 6836.73 | 6858.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:45:00 | 6814.85 | 6786.45 | 6801.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 6485.79 | 6576.09 | 6622.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 6485.65 | 6576.09 | 6622.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 6474.11 | 6576.09 | 6622.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 13:15:00 | 6272.10 | 6221.49 | 6298.94 | SL hit (close>ema200) qty=0.50 sl=6221.49 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 6113.10 | 6063.07 | 6061.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 6183.35 | 6102.14 | 6081.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 6240.15 | 6287.39 | 6229.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 6240.15 | 6287.39 | 6229.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 6240.15 | 6287.39 | 6229.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 6194.65 | 6287.39 | 6229.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 6271.60 | 6284.23 | 6233.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:30:00 | 6243.70 | 6284.23 | 6233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 6607.95 | 6364.16 | 6288.94 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 6409.20 | 6463.61 | 6469.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6329.30 | 6416.35 | 6442.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 6125.55 | 6118.50 | 6196.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:30:00 | 6160.00 | 6118.50 | 6196.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 5945.45 | 6026.03 | 6111.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:00:00 | 5919.80 | 5991.80 | 6080.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:15:00 | 5919.95 | 5981.32 | 6067.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 5886.00 | 5925.48 | 6003.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:15:00 | 5920.10 | 5857.45 | 5915.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 5910.00 | 5867.96 | 5914.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:30:00 | 5940.55 | 5867.96 | 5914.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 5936.20 | 5881.61 | 5916.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 5936.20 | 5881.61 | 5916.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 5924.05 | 5890.09 | 5917.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 5943.85 | 5890.09 | 5917.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 5854.85 | 5782.28 | 5817.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 5854.85 | 5782.28 | 5817.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 5893.60 | 5804.54 | 5824.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 5893.60 | 5804.54 | 5824.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 5825.10 | 5808.65 | 5824.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 5802.00 | 5811.12 | 5824.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 5809.40 | 5818.07 | 5826.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:00:00 | 5810.00 | 5825.56 | 5828.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 5865.70 | 5833.58 | 5831.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 5865.70 | 5833.58 | 5831.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 5876.65 | 5842.20 | 5835.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 5840.65 | 5851.44 | 5842.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 14:15:00 | 5840.65 | 5851.44 | 5842.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 5840.65 | 5851.44 | 5842.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 5840.65 | 5851.44 | 5842.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 5847.00 | 5850.55 | 5842.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 5675.10 | 5850.55 | 5842.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 5621.85 | 5804.81 | 5822.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 5589.00 | 5732.78 | 5785.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 15:15:00 | 5718.40 | 5693.13 | 5745.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:15:00 | 5585.80 | 5693.13 | 5745.64 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 12:15:00 | 5652.45 | 5628.68 | 5696.93 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 5693.70 | 5641.69 | 5696.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 5693.70 | 5641.69 | 5696.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 5663.40 | 5646.03 | 5693.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 5641.10 | 5646.53 | 5689.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 6053.00 | 5726.96 | 5718.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 125 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 6053.00 | 5726.96 | 5718.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 6265.00 | 5890.57 | 5797.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 6190.50 | 6220.11 | 6076.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 6203.45 | 6220.11 | 6076.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 6267.90 | 6270.75 | 6207.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 6251.15 | 6270.75 | 6207.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 6224.95 | 6261.59 | 6209.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 6224.95 | 6261.59 | 6209.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 6267.70 | 6262.81 | 6214.76 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 6169.90 | 6199.31 | 6202.45 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 6252.50 | 6205.83 | 6204.20 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 6148.05 | 6192.65 | 6198.61 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 6249.00 | 6205.65 | 6202.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 6262.70 | 6217.06 | 6208.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 6270.75 | 6281.78 | 6251.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 6270.75 | 6281.78 | 6251.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 6270.75 | 6281.78 | 6251.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 6303.70 | 6281.78 | 6251.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 6241.00 | 6269.94 | 6255.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 6244.85 | 6269.94 | 6255.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 6283.05 | 6272.56 | 6257.95 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 6133.05 | 6235.21 | 6246.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 6020.85 | 6180.67 | 6218.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 5972.00 | 5971.09 | 6039.85 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 5871.05 | 5971.09 | 6039.85 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 5577.50 | 5685.77 | 5792.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 5607.40 | 5541.92 | 5617.19 | SL hit (close>ema200) qty=0.50 sl=5541.92 alert=retest1 |

### Cycle 131 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 5787.50 | 5629.86 | 5610.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 5813.50 | 5666.59 | 5629.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 6119.00 | 6133.08 | 5961.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:15:00 | 6228.00 | 6133.08 | 5961.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 13:45:00 | 6179.25 | 6163.18 | 6043.05 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 6024.85 | 6132.24 | 6058.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 6024.85 | 6132.24 | 6058.67 | SL hit (close<ema400) qty=1.00 sl=6058.67 alert=retest1 |

### Cycle 132 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 6089.55 | 6205.70 | 6210.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 6042.00 | 6138.06 | 6175.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 5988.00 | 5969.91 | 6051.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 5976.80 | 5969.91 | 6051.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 5881.85 | 5952.29 | 6036.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 5865.95 | 5928.24 | 6017.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 14:15:00 | 5875.05 | 5887.71 | 5973.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 14:45:00 | 5873.60 | 5878.15 | 5961.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 6039.40 | 5978.36 | 5973.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 6039.40 | 5978.36 | 5973.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 6102.85 | 6003.26 | 5984.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 6098.80 | 6102.97 | 6051.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 6098.80 | 6102.97 | 6051.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 6050.00 | 6088.86 | 6061.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 6100.00 | 6088.86 | 6061.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 6144.00 | 6099.89 | 6069.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 6198.45 | 6115.54 | 6079.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 6043.70 | 6121.55 | 6127.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 6043.70 | 6121.55 | 6127.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 6001.20 | 6071.67 | 6101.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 6065.00 | 6027.47 | 6067.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 6065.00 | 6027.47 | 6067.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 6065.00 | 6027.47 | 6067.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 6107.55 | 6027.47 | 6067.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 6072.30 | 6036.44 | 6067.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 6100.90 | 6036.44 | 6067.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 6107.30 | 6062.41 | 6074.66 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 6135.90 | 6084.00 | 6082.78 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 6058.55 | 6079.87 | 6081.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 6041.00 | 6068.92 | 6075.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 12:15:00 | 6089.35 | 6073.00 | 6077.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 12:15:00 | 6089.35 | 6073.00 | 6077.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 6089.35 | 6073.00 | 6077.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 6089.35 | 6073.00 | 6077.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 6100.50 | 6078.50 | 6079.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:30:00 | 6134.65 | 6078.50 | 6079.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 6036.25 | 6070.05 | 6075.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 6005.00 | 6070.05 | 6075.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 6115.00 | 6068.63 | 6073.05 | SL hit (close>static) qty=1.00 sl=6100.50 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 6135.80 | 6082.07 | 6078.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 6299.15 | 6163.30 | 6131.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 12:15:00 | 6321.00 | 6331.70 | 6262.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 12:45:00 | 6303.50 | 6331.70 | 6262.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 6462.00 | 6480.90 | 6428.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 6416.10 | 6480.90 | 6428.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 6316.05 | 6447.93 | 6418.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 6316.05 | 6447.93 | 6418.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 6303.50 | 6419.04 | 6407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 6305.05 | 6419.04 | 6407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 6319.60 | 6399.16 | 6399.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 6285.75 | 6331.50 | 6360.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 6196.00 | 6191.72 | 6252.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 6196.00 | 6191.72 | 6252.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 6194.35 | 6192.24 | 6246.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 6247.65 | 6192.24 | 6246.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 6161.20 | 6186.03 | 6239.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 6240.00 | 6186.03 | 6239.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 6231.50 | 6189.84 | 6231.30 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 6253.75 | 6222.15 | 6219.17 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 13:15:00 | 6176.00 | 6211.49 | 6215.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 14:15:00 | 6171.80 | 6203.55 | 6211.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 5970.00 | 5841.85 | 5956.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 13:15:00 | 5970.00 | 5841.85 | 5956.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 5970.00 | 5841.85 | 5956.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:00:00 | 5970.00 | 5841.85 | 5956.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 5956.80 | 5864.84 | 5956.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:45:00 | 5940.00 | 5864.84 | 5956.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 5888.05 | 5869.48 | 5950.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 5994.70 | 5869.48 | 5950.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 5944.90 | 5884.56 | 5950.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 5918.25 | 5932.15 | 5953.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 5860.00 | 5934.16 | 5952.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:00:00 | 5919.05 | 5900.48 | 5919.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:30:00 | 5912.25 | 5910.72 | 5922.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 5900.55 | 5908.69 | 5920.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 6067.00 | 5937.99 | 5929.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 6067.00 | 5937.99 | 5929.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 6071.50 | 5987.56 | 5955.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 6236.00 | 6236.82 | 6167.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 6309.50 | 6236.25 | 6173.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:15:00 | 6624.98 | 6439.78 | 6328.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 6505.00 | 6509.33 | 6411.39 | SL hit (close<ema200) qty=0.50 sl=6509.33 alert=retest1 |

### Cycle 142 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 6381.00 | 6468.41 | 6471.02 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 6510.00 | 6436.83 | 6434.43 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 6378.50 | 6448.56 | 6449.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 12:15:00 | 6340.50 | 6414.14 | 6432.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 6414.50 | 6384.10 | 6409.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 6414.50 | 6384.10 | 6409.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 6414.50 | 6384.10 | 6409.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 6421.50 | 6384.10 | 6409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 6344.00 | 6376.08 | 6403.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 6315.00 | 6363.26 | 6395.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 6318.00 | 6358.61 | 6390.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 6476.50 | 6376.78 | 6373.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 6476.50 | 6376.78 | 6373.08 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 6316.00 | 6371.22 | 6371.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 6199.00 | 6325.29 | 6349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 6088.50 | 6049.87 | 6112.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 6088.50 | 6049.87 | 6112.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6356.00 | 6116.80 | 6132.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 6398.50 | 6116.80 | 6132.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 6381.00 | 6169.64 | 6155.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 6475.00 | 6377.04 | 6304.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 7235.00 | 7247.10 | 7118.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 7235.00 | 7247.10 | 7118.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 7224.00 | 7249.41 | 7162.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:30:00 | 7228.50 | 7243.53 | 7167.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:30:00 | 7225.50 | 7231.12 | 7168.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 7106.00 | 7206.10 | 7163.26 | SL hit (close<static) qty=1.00 sl=7150.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 7112.50 | 7138.65 | 7140.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 7051.00 | 7121.12 | 7132.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 7085.00 | 7042.59 | 7076.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 7026.00 | 7039.27 | 7071.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 7080.00 | 7039.27 | 7071.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 7053.00 | 7042.02 | 7069.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 7055.50 | 7042.02 | 7069.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 7040.00 | 7043.43 | 7063.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 7040.00 | 7043.43 | 7063.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 7035.00 | 7041.75 | 7061.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 7371.00 | 7041.75 | 7061.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 7512.00 | 7135.80 | 7102.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 7679.00 | 7244.44 | 7154.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 14:15:00 | 7505.00 | 7511.10 | 7404.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 7505.00 | 7511.10 | 7404.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 7521.00 | 7553.94 | 7525.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 7512.50 | 7553.94 | 7525.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 7535.50 | 7550.25 | 7526.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 7616.50 | 7556.70 | 7531.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 7480.00 | 7536.14 | 7527.97 | SL hit (close<static) qty=1.00 sl=7522.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 7500.00 | 7521.27 | 7522.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 7473.00 | 7511.62 | 7518.06 | Break + close below crossover candle low |

### Cycle 151 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 7596.50 | 7528.59 | 7525.19 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 7474.00 | 7524.36 | 7528.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 7450.00 | 7509.48 | 7521.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 7565.00 | 7520.59 | 7525.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 7515.00 | 7519.47 | 7524.74 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 7555.00 | 7530.27 | 7528.21 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 7490.00 | 7522.99 | 7526.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 7470.50 | 7504.80 | 7516.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 7478.50 | 7479.55 | 7496.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 7472.50 | 7478.65 | 7492.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 7480.00 | 7478.92 | 7491.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:45:00 | 7472.50 | 7477.24 | 7489.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 7478.50 | 7477.49 | 7488.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 7478.50 | 7477.49 | 7488.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 7480.00 | 7477.99 | 7488.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 7484.50 | 7477.99 | 7488.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 7456.00 | 7473.59 | 7485.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 10:45:00 | 7420.50 | 7461.48 | 7478.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:45:00 | 7435.00 | 7393.67 | 7418.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 7425.00 | 7393.67 | 7418.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7104.57 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7098.88 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7106.00 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7098.88 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7049.47 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7063.25 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7053.75 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 6694.00 | 6646.16 | 6644.42 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 6574.50 | 6638.26 | 6645.43 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 6670.50 | 6624.05 | 6622.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 6686.00 | 6644.43 | 6632.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 6620.50 | 6641.58 | 6633.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 6620.00 | 6637.26 | 6632.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 6625.00 | 6631.41 | 6629.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 6595.50 | 6624.23 | 6626.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 6595.50 | 6624.23 | 6626.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 6585.50 | 6607.28 | 6617.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 6609.00 | 6602.07 | 6612.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 6616.50 | 6604.95 | 6612.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 6626.00 | 6604.95 | 6612.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 6607.00 | 6605.36 | 6612.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 6712.00 | 6605.36 | 6612.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 6638.00 | 6611.89 | 6614.41 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 6670.00 | 6623.51 | 6619.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 6757.00 | 6675.80 | 6649.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 6802.50 | 6803.12 | 6747.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:45:00 | 6812.50 | 6803.12 | 6747.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 6741.00 | 6783.55 | 6758.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 6741.00 | 6783.55 | 6758.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 6737.50 | 6774.34 | 6756.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 6740.00 | 6774.34 | 6756.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 6741.00 | 6767.67 | 6755.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 6723.50 | 6767.67 | 6755.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 6772.00 | 6775.00 | 6762.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 6772.00 | 6775.00 | 6762.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 6778.50 | 6775.70 | 6763.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 6808.00 | 6775.70 | 6763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 6795.00 | 6779.56 | 6766.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 6844.00 | 6815.33 | 6797.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 6869.00 | 6816.61 | 6801.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:00:00 | 6844.00 | 6822.55 | 6806.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 6840.00 | 6828.03 | 6812.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 6840.50 | 6829.16 | 6815.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 6811.00 | 6829.16 | 6815.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 6812.50 | 6829.00 | 6817.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 6824.00 | 6829.00 | 6817.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 6800.00 | 6823.20 | 6816.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 6802.00 | 6823.20 | 6816.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 6804.50 | 6819.46 | 6815.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 6790.50 | 6819.46 | 6815.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 6756.00 | 6796.33 | 6804.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 6704.50 | 6704.22 | 6735.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:30:00 | 6720.00 | 6704.22 | 6735.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 6688.50 | 6686.60 | 6713.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 6690.00 | 6686.60 | 6713.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 6716.50 | 6692.58 | 6713.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 6699.00 | 6692.58 | 6713.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 6680.50 | 6690.16 | 6710.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 6675.00 | 6690.16 | 6710.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:00:00 | 6675.00 | 6611.46 | 6632.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 6570.00 | 6536.65 | 6535.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 6570.00 | 6536.65 | 6535.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6610.50 | 6551.42 | 6542.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 6615.00 | 6589.03 | 6575.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 6587.00 | 6582.43 | 6578.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 6561.00 | 6574.24 | 6574.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 6561.00 | 6574.24 | 6574.85 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 6588.00 | 6576.99 | 6576.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 15:15:00 | 6600.50 | 6585.49 | 6580.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 6548.50 | 6578.09 | 6577.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 6536.00 | 6569.67 | 6573.80 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 6609.50 | 6577.64 | 6577.04 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 6535.00 | 6579.86 | 6580.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6498.00 | 6558.63 | 6570.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 6547.00 | 6454.25 | 6483.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 6544.00 | 6472.20 | 6489.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 6401.00 | 6472.20 | 6489.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 6276.50 | 6241.59 | 6241.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 6276.50 | 6241.59 | 6241.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 6392.00 | 6271.67 | 6254.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 14:15:00 | 6400.00 | 6405.19 | 6346.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 6400.00 | 6405.19 | 6346.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 6370.00 | 6395.24 | 6370.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 6370.00 | 6395.24 | 6370.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 6353.00 | 6386.79 | 6368.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 6389.00 | 6383.64 | 6368.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 6399.00 | 6375.57 | 6370.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 6326.00 | 6365.66 | 6366.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 6326.00 | 6365.66 | 6366.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 6321.00 | 6345.72 | 6356.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 6288.50 | 6317.55 | 6334.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 6279.50 | 6317.55 | 6334.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 6295.50 | 6307.83 | 6326.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 6222.00 | 6307.57 | 6324.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | SL hit (close>static) qty=1.00 sl=6343.50 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 6559.00 | 6354.14 | 6338.39 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 6355.00 | 6366.57 | 6368.04 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 6401.00 | 6370.88 | 6369.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 6406.00 | 6383.42 | 6376.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 6341.50 | 6376.14 | 6373.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 6358.50 | 6372.61 | 6372.23 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 6360.00 | 6370.92 | 6371.59 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 6387.00 | 6374.14 | 6372.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 6405.00 | 6380.31 | 6375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 6367.00 | 6387.77 | 6381.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 6408.00 | 6391.82 | 6384.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 6429.00 | 6398.44 | 6390.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 6460.50 | 6412.60 | 6398.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 6430.00 | 6434.48 | 6423.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 6435.00 | 6429.15 | 6424.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 6415.50 | 6426.42 | 6423.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 6415.50 | 6426.42 | 6423.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 6456.00 | 6422.47 | 6421.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 6487.00 | 6435.37 | 6427.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 6470.50 | 6474.87 | 6453.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:45:00 | 6468.00 | 6474.87 | 6453.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 6460.00 | 6471.89 | 6454.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 6453.50 | 6471.89 | 6454.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 6456.00 | 6468.71 | 6454.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 6445.50 | 6468.71 | 6454.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 6445.50 | 6464.07 | 6453.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 6489.00 | 6465.96 | 6455.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 6474.00 | 6465.41 | 6456.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 6474.00 | 6466.34 | 6458.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 6474.00 | 6466.34 | 6458.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 6446.00 | 6464.22 | 6459.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 6511.00 | 6464.22 | 6459.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 6442.50 | 6473.30 | 6466.11 | SL hit (close<static) qty=1.00 sl=6446.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 6414.00 | 6455.63 | 6458.96 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 6493.00 | 6460.06 | 6459.45 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 6439.50 | 6457.62 | 6459.09 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 6528.00 | 6464.52 | 6460.50 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 6440.00 | 6489.61 | 6493.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 6434.00 | 6468.01 | 6479.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 6474.50 | 6417.94 | 6412.68 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 6393.00 | 6412.99 | 6414.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 6376.50 | 6405.69 | 6411.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 6381.00 | 6380.52 | 6392.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 6381.00 | 6380.52 | 6392.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 6352.50 | 6371.87 | 6386.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 6315.00 | 6362.50 | 6380.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:00:00 | 6306.50 | 6330.21 | 6357.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 6285.00 | 6330.67 | 6355.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 6299.00 | 6324.33 | 6350.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 6314.00 | 6258.80 | 6271.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 6314.00 | 6258.80 | 6271.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 6330.00 | 6273.04 | 6276.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 6296.00 | 6273.04 | 6276.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 6225.00 | 6190.82 | 6174.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 6195.00 | 6195.92 | 6180.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 6195.00 | 6195.92 | 6180.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 6189.00 | 6194.54 | 6181.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 6191.00 | 6194.54 | 6181.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 6173.00 | 6190.23 | 6180.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 6220.00 | 6181.15 | 6179.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 6230.00 | 6184.92 | 6181.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 6206.50 | 6188.17 | 6184.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 6229.00 | 6194.23 | 6188.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6188.00 | 6192.98 | 6188.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 6188.00 | 6192.98 | 6188.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6188.00 | 6191.99 | 6188.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 6210.00 | 6187.88 | 6186.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 6197.00 | 6210.82 | 6203.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 6150.00 | 6180.36 | 6190.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 6130.50 | 6120.90 | 6143.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 6135.00 | 6123.72 | 6142.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 6105.50 | 6123.72 | 6142.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6103.00 | 6119.58 | 6138.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 6089.00 | 6111.02 | 6125.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:30:00 | 6080.00 | 6100.78 | 6115.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 6082.50 | 6096.92 | 6110.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 6082.00 | 6095.63 | 6108.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 6095.00 | 6091.40 | 6103.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 6117.00 | 6091.40 | 6103.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 6099.00 | 6088.50 | 6097.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 6078.00 | 6094.10 | 6099.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 6086.00 | 6092.48 | 6097.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 6085.00 | 6085.58 | 6094.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 6089.00 | 6075.96 | 6083.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 6080.00 | 6076.77 | 6082.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 6070.00 | 6076.77 | 6082.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5784.55 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5776.00 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5778.38 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5777.90 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5774.10 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5781.70 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5780.75 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5784.55 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5766.50 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 5870.50 | 5747.84 | 5746.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 5970.00 | 5792.28 | 5767.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 5972.00 | 6082.06 | 6001.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 5953.00 | 6056.25 | 5997.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 5953.00 | 6056.25 | 5997.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 5978.00 | 6030.16 | 5995.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 5974.50 | 6030.16 | 5995.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5940.00 | 5996.29 | 5986.90 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 5875.00 | 5972.03 | 5976.73 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 5952.00 | 5928.33 | 5926.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 5992.50 | 5941.17 | 5932.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5983.50 | 6008.43 | 5985.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5972.50 | 6001.24 | 5984.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 5972.50 | 6001.24 | 5984.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 5927.00 | 5986.39 | 5979.40 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 5934.00 | 5970.97 | 5973.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 5890.00 | 5951.98 | 5964.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 5780.00 | 5777.74 | 5826.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 5780.00 | 5777.74 | 5826.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 5846.00 | 5794.61 | 5822.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 5846.00 | 5794.61 | 5822.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 5805.00 | 5796.68 | 5820.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 5799.00 | 5802.08 | 5818.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 5781.50 | 5804.56 | 5818.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 5787.00 | 5804.56 | 5818.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 5812.00 | 5850.62 | 5852.78 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 5864.00 | 5855.77 | 5854.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 5885.00 | 5861.62 | 5857.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 5896.50 | 5915.47 | 5894.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 5896.50 | 5915.47 | 5894.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 5881.00 | 5908.58 | 5893.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 5870.50 | 5908.58 | 5893.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 5840.00 | 5894.86 | 5888.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 5916.00 | 5894.86 | 5888.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 5899.50 | 5895.79 | 5889.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 5952.00 | 6036.77 | 6038.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 5952.00 | 6036.77 | 6038.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 5925.00 | 5982.27 | 6006.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 5929.00 | 5926.75 | 5954.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 5921.50 | 5926.75 | 5954.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 5923.00 | 5918.26 | 5943.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 5900.50 | 5917.61 | 5940.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:15:00 | 5910.00 | 5911.92 | 5931.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 5910.00 | 5912.44 | 5930.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 5910.50 | 5916.43 | 5927.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 5880.00 | 5892.68 | 5909.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 5859.00 | 5884.55 | 5904.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 5855.00 | 5879.04 | 5900.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 5848.00 | 5876.53 | 5897.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5953.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 5984.50 | 5907.26 | 5897.04 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 5902.50 | 5918.72 | 5918.99 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 5950.00 | 5920.47 | 5919.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 5959.50 | 5929.40 | 5923.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 5907.50 | 5923.31 | 5923.41 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 5937.00 | 5921.86 | 5921.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5955.00 | 5928.49 | 5924.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 5913.00 | 5932.03 | 5927.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 5911.50 | 5927.93 | 5926.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 5912.00 | 5927.93 | 5926.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5921.00 | 5926.54 | 5925.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 5921.00 | 5926.54 | 5925.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 5910.00 | 5923.23 | 5924.45 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 5956.50 | 5923.52 | 5923.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 15:15:00 | 5970.00 | 5941.13 | 5932.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 5951.50 | 5964.66 | 5952.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 5931.00 | 5957.93 | 5950.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 5931.00 | 5957.93 | 5950.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 5910.00 | 5948.34 | 5946.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 5910.00 | 5948.34 | 5946.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 5895.00 | 5937.67 | 5942.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 5886.50 | 5924.93 | 5935.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 5980.50 | 5924.19 | 5931.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 5933.50 | 5926.05 | 5931.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 5918.50 | 5927.30 | 5931.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 5908.50 | 5927.30 | 5931.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6197.00 | 5978.47 | 5953.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 09:15:00 | 6197.00 | 5978.47 | 5953.60 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 6005.00 | 6069.72 | 6069.88 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 6099.00 | 6075.58 | 6072.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 6124.50 | 6092.81 | 6081.78 | Break + close above crossover candle high |

### Cycle 204 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5994.50 | 6075.42 | 6075.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5960.50 | 6007.90 | 6038.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 5919.00 | 5913.09 | 5961.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 5927.50 | 5913.09 | 5961.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5964.50 | 5916.72 | 5950.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 5964.50 | 5916.72 | 5950.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 5975.00 | 5928.38 | 5952.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 5960.00 | 5928.38 | 5952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 5983.50 | 5948.30 | 5958.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:30:00 | 5995.50 | 5948.30 | 5958.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 5924.00 | 5949.95 | 5957.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 5911.00 | 5949.95 | 5957.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 5914.50 | 5887.65 | 5886.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 5914.50 | 5887.65 | 5886.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 5975.00 | 5911.82 | 5898.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 5890.00 | 5907.46 | 5897.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5861.50 | 5898.26 | 5894.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 5853.00 | 5898.26 | 5894.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 5892.00 | 5897.01 | 5893.96 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 5862.00 | 5890.01 | 5891.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 5840.00 | 5878.41 | 5885.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 5886.00 | 5873.46 | 5881.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 5888.50 | 5876.47 | 5882.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 5894.50 | 5876.47 | 5882.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 5913.50 | 5883.87 | 5885.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 5913.50 | 5883.87 | 5885.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 5941.50 | 5895.40 | 5890.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 6037.00 | 5923.72 | 5903.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 5950.50 | 5962.60 | 5934.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 5950.50 | 5962.60 | 5934.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 5960.00 | 5962.08 | 5936.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 5957.50 | 5962.08 | 5936.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 5915.00 | 5951.93 | 5936.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 5915.00 | 5951.93 | 5936.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 5905.00 | 5942.54 | 5933.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 5896.00 | 5942.54 | 5933.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 5859.00 | 5925.84 | 5926.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 5821.00 | 5904.87 | 5917.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 5931.00 | 5891.43 | 5904.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5950.00 | 5903.14 | 5909.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 5971.50 | 5903.14 | 5909.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 5968.50 | 5916.21 | 5914.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 6013.50 | 5982.35 | 5968.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 6501.50 | 6511.58 | 6374.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 6485.00 | 6511.58 | 6374.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 6390.00 | 6472.99 | 6421.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 6365.50 | 6472.99 | 6421.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 6365.00 | 6451.39 | 6416.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 6365.00 | 6451.39 | 6416.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 6429.50 | 6432.34 | 6417.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 6550.00 | 6432.34 | 6417.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 6749.00 | 6807.14 | 6809.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 6749.00 | 6807.14 | 6809.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 6729.00 | 6791.51 | 6802.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 6761.00 | 6726.54 | 6722.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 6919.00 | 6766.07 | 6740.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 6811.50 | 6826.08 | 6786.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 6811.50 | 6826.08 | 6786.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 6770.00 | 6811.73 | 6786.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 6770.00 | 6811.73 | 6786.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 6780.00 | 6805.39 | 6786.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:45:00 | 6789.50 | 6789.18 | 6781.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:30:00 | 6800.50 | 6787.35 | 6781.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 6736.50 | 6777.18 | 6777.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 6736.50 | 6777.18 | 6777.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 6639.00 | 6749.54 | 6764.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 6669.00 | 6667.44 | 6707.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 6669.00 | 6667.44 | 6707.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 6660.00 | 6656.61 | 6688.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 6641.00 | 6659.51 | 6684.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 6760.00 | 6687.49 | 6692.21 | SL hit (close>static) qty=1.00 sl=6726.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6744.00 | 6700.00 | 6697.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 6801.00 | 6743.29 | 6719.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 6765.00 | 6770.92 | 6740.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 6622.50 | 6770.92 | 6740.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6595.00 | 6735.74 | 6727.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 6592.00 | 6735.74 | 6727.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 6615.00 | 6711.59 | 6717.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 6584.50 | 6651.78 | 6685.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 6664.50 | 6635.57 | 6667.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 6630.00 | 6634.46 | 6664.51 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 6807.00 | 6684.92 | 6677.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 7073.00 | 6811.63 | 6751.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7010.00 | 7018.10 | 6892.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 7010.00 | 7018.10 | 6892.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 6980.00 | 7012.84 | 6930.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 6925.00 | 7012.84 | 6930.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 6929.00 | 6987.30 | 6932.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 6990.50 | 6987.30 | 6932.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 7050.00 | 6999.84 | 6943.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 7246.00 | 7146.48 | 7058.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 7080.00 | 7183.94 | 7187.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 7080.00 | 7183.94 | 7187.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 6992.00 | 7145.55 | 7169.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 7066.00 | 7132.00 | 7155.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 7056.00 | 7102.04 | 7136.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 6712.70 | 6927.50 | 7039.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 6703.20 | 6927.50 | 7039.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 6962.50 | 6835.48 | 6927.92 | SL hit (close>ema200) qty=0.50 sl=6835.48 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 7415.00 | 7046.87 | 7004.89 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 7043.50 | 7174.14 | 7186.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 6832.00 | 7046.02 | 7119.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 7000.00 | 6968.75 | 7060.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:15:00 | 7109.50 | 6968.75 | 7060.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 6960.00 | 6967.00 | 7051.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 6870.00 | 6967.00 | 7051.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 6947.00 | 6971.52 | 7025.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 6945.50 | 6982.82 | 7025.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 7298.50 | 7107.79 | 7064.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 7146.00 | 7168.41 | 7125.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 7146.00 | 7168.41 | 7125.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 7150.00 | 7164.72 | 7127.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 7169.00 | 7164.72 | 7127.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 7160.50 | 7163.88 | 7130.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 7229.00 | 7154.65 | 7136.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 14:30:00 | 7235.00 | 7154.57 | 7142.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 7243.00 | 7159.66 | 7145.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 7214.00 | 7177.90 | 7156.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 7190.50 | 7187.30 | 7167.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 7160.00 | 7187.30 | 7167.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 7194.50 | 7188.74 | 7169.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 7178.00 | 7188.74 | 7169.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 7161.00 | 7183.19 | 7169.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 7336.00 | 7183.19 | 7169.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 7238.00 | 7208.85 | 7195.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 7140.00 | 7221.83 | 7234.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 7192.00 | 7101.24 | 7128.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 7164.50 | 7113.89 | 7131.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 7206.50 | 7113.89 | 7131.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 7235.00 | 7153.65 | 7147.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 7274.00 | 7186.66 | 7164.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 7223.00 | 7237.72 | 7208.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 7264.00 | 7242.98 | 7213.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 7271.50 | 7242.98 | 7213.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 7199.00 | 7248.43 | 7231.19 | SL hit (close<static) qty=1.00 sl=7211.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 7209.00 | 7227.41 | 7228.51 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 7259.00 | 7233.73 | 7231.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 11:15:00 | 7272.50 | 7241.48 | 7235.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 7236.50 | 7242.71 | 7236.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 7233.00 | 7240.77 | 7236.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 7230.00 | 7240.77 | 7236.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 7230.00 | 7238.61 | 7235.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 7262.00 | 7238.61 | 7235.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 7245.00 | 7239.89 | 7236.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-07 10:15:00 | 7988.20 | 7653.55 | 7549.66 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 11:45:00 | 7150.00 | 2024-04-24 09:15:00 | 7865.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-05-17 09:15:00 | 9238.25 | 2024-05-21 09:15:00 | 9700.16 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-05-17 09:15:00 | 9238.25 | 2024-05-22 09:15:00 | 9620.00 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2024-05-27 13:30:00 | 8807.70 | 2024-05-28 15:15:00 | 8367.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 13:30:00 | 8807.70 | 2024-05-29 11:15:00 | 8495.30 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2024-06-03 09:15:00 | 8968.75 | 2024-06-04 10:15:00 | 8290.45 | STOP_HIT | 1.00 | -7.56% |
| BUY | retest2 | 2024-06-03 11:00:00 | 8774.00 | 2024-06-04 10:15:00 | 8290.45 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest2 | 2024-06-03 11:30:00 | 8771.00 | 2024-06-04 10:15:00 | 8290.45 | STOP_HIT | 1.00 | -5.48% |
| BUY | retest2 | 2024-06-03 12:15:00 | 8774.50 | 2024-06-04 10:15:00 | 8290.45 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2024-06-14 11:30:00 | 9259.55 | 2024-06-19 09:15:00 | 9055.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-06-18 09:15:00 | 9530.05 | 2024-06-19 09:15:00 | 9055.00 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest2 | 2024-06-18 15:15:00 | 9275.00 | 2024-06-19 09:15:00 | 9055.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-06-25 11:15:00 | 8360.00 | 2024-07-01 09:15:00 | 8423.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-25 12:45:00 | 8335.85 | 2024-07-01 09:15:00 | 8423.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-06-28 10:15:00 | 8286.25 | 2024-07-01 09:15:00 | 8423.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-07-08 10:15:00 | 9263.80 | 2024-07-10 09:15:00 | 8879.95 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2024-07-08 12:30:00 | 9264.65 | 2024-07-10 09:15:00 | 8879.95 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-07-08 14:15:00 | 9253.80 | 2024-07-10 09:15:00 | 8879.95 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2024-08-06 12:30:00 | 7798.30 | 2024-08-12 09:15:00 | 7408.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 15:00:00 | 7798.15 | 2024-08-12 09:15:00 | 7408.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 12:30:00 | 7798.30 | 2024-08-12 12:15:00 | 7617.75 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2024-08-06 15:00:00 | 7798.15 | 2024-08-12 12:15:00 | 7617.75 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2024-08-08 09:30:00 | 7767.85 | 2024-08-14 09:15:00 | 7379.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:30:00 | 7767.85 | 2024-08-16 12:15:00 | 7240.00 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2024-08-28 10:30:00 | 7269.95 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-28 11:30:00 | 7269.95 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-28 13:15:00 | 7272.00 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-28 15:15:00 | 7260.00 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-08-29 14:15:00 | 7197.30 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-08-29 15:15:00 | 7191.00 | 2024-08-30 12:15:00 | 7341.15 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-09-04 12:30:00 | 7193.20 | 2024-09-05 09:15:00 | 7526.80 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2024-09-04 14:00:00 | 7203.90 | 2024-09-05 09:15:00 | 7526.80 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-09-13 11:30:00 | 7800.75 | 2024-09-23 11:15:00 | 8580.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-10-07 10:45:00 | 7958.90 | 2024-10-08 14:15:00 | 8171.05 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest1 | 2024-10-10 15:00:00 | 8322.30 | 2024-10-11 11:15:00 | 8204.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-10-24 12:15:00 | 7938.50 | 2024-10-28 09:15:00 | 7541.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:45:00 | 7941.00 | 2024-10-28 09:15:00 | 7543.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:15:00 | 7930.00 | 2024-10-28 09:15:00 | 7533.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 7875.65 | 2024-10-28 09:15:00 | 7481.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:15:00 | 7938.50 | 2024-10-29 15:15:00 | 7570.00 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2024-10-24 12:45:00 | 7941.00 | 2024-10-29 15:15:00 | 7570.00 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2024-10-24 14:15:00 | 7930.00 | 2024-10-29 15:15:00 | 7570.00 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2024-10-25 09:15:00 | 7875.65 | 2024-10-29 15:15:00 | 7570.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2024-10-25 10:45:00 | 7774.30 | 2024-10-31 11:15:00 | 7653.10 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-10-25 13:00:00 | 7788.30 | 2024-10-31 11:15:00 | 7653.10 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2024-10-25 13:45:00 | 7770.00 | 2024-10-31 11:15:00 | 7653.10 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-11-05 09:15:00 | 7773.25 | 2024-11-08 09:15:00 | 7648.85 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-11-05 10:30:00 | 7831.75 | 2024-11-08 09:15:00 | 7648.85 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-11-07 15:00:00 | 7795.15 | 2024-11-08 09:15:00 | 7648.85 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-11-18 10:45:00 | 6635.00 | 2024-11-25 10:15:00 | 6733.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-11-18 11:30:00 | 6630.10 | 2024-11-25 10:15:00 | 6733.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-11-18 12:45:00 | 6626.00 | 2024-11-25 10:15:00 | 6733.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-11-19 15:00:00 | 6629.40 | 2024-11-25 10:15:00 | 6733.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-11-29 14:15:00 | 6942.70 | 2024-12-04 13:15:00 | 6902.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-02 12:00:00 | 6979.00 | 2024-12-04 13:15:00 | 6902.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-12-09 10:15:00 | 6827.15 | 2024-12-19 09:15:00 | 6485.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 11:45:00 | 6827.00 | 2024-12-19 09:15:00 | 6485.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:45:00 | 6814.85 | 2024-12-19 09:15:00 | 6474.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 10:15:00 | 6827.15 | 2024-12-24 13:15:00 | 6272.10 | STOP_HIT | 0.50 | 8.13% |
| SELL | retest2 | 2024-12-09 11:45:00 | 6827.00 | 2024-12-24 13:15:00 | 6272.10 | STOP_HIT | 0.50 | 8.13% |
| SELL | retest2 | 2024-12-11 12:45:00 | 6814.85 | 2024-12-24 13:15:00 | 6272.10 | STOP_HIT | 0.50 | 7.96% |
| SELL | retest2 | 2025-01-16 12:00:00 | 5919.80 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-01-16 13:15:00 | 5919.95 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-01-17 10:45:00 | 5886.00 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-01-20 12:15:00 | 5920.10 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-01-23 13:15:00 | 5802.00 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-23 14:15:00 | 5809.40 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-01-24 10:00:00 | 5810.00 | 2025-01-24 10:15:00 | 5865.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest1 | 2025-01-28 09:15:00 | 5585.80 | 2025-01-29 09:15:00 | 6053.00 | STOP_HIT | 1.00 | -8.36% |
| SELL | retest1 | 2025-01-28 12:15:00 | 5652.45 | 2025-01-29 09:15:00 | 6053.00 | STOP_HIT | 1.00 | -7.09% |
| SELL | retest2 | 2025-01-28 15:15:00 | 5641.10 | 2025-01-29 09:15:00 | 6053.00 | STOP_HIT | 1.00 | -7.30% |
| SELL | retest1 | 2025-02-12 09:15:00 | 5871.05 | 2025-02-14 10:15:00 | 5577.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-12 09:15:00 | 5871.05 | 2025-02-17 14:15:00 | 5607.40 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2025-02-18 09:15:00 | 5500.00 | 2025-02-19 10:15:00 | 5735.00 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-02-18 10:00:00 | 5532.50 | 2025-02-19 10:15:00 | 5735.00 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2025-02-21 09:15:00 | 6228.00 | 2025-02-24 09:15:00 | 6024.85 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2025-02-21 13:45:00 | 6179.25 | 2025-02-24 09:15:00 | 6024.85 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-02-24 11:15:00 | 6136.00 | 2025-02-27 12:15:00 | 6089.55 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-02-27 10:45:00 | 6120.50 | 2025-02-27 12:15:00 | 6089.55 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-03-03 10:30:00 | 5865.95 | 2025-03-05 11:15:00 | 6039.40 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-03-03 14:15:00 | 5875.05 | 2025-03-05 11:15:00 | 6039.40 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-03-03 14:45:00 | 5873.60 | 2025-03-05 11:15:00 | 6039.40 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-03-07 10:30:00 | 6198.45 | 2025-03-11 09:15:00 | 6043.70 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-13 15:15:00 | 6005.00 | 2025-03-17 09:15:00 | 6115.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-08 15:00:00 | 5918.25 | 2025-04-15 09:15:00 | 6067.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-04-09 09:15:00 | 5860.00 | 2025-04-15 09:15:00 | 6067.00 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-04-11 10:00:00 | 5919.05 | 2025-04-15 09:15:00 | 6067.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-04-11 10:30:00 | 5912.25 | 2025-04-15 09:15:00 | 6067.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest1 | 2025-04-21 09:15:00 | 6309.50 | 2025-04-22 09:15:00 | 6624.98 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 09:15:00 | 6309.50 | 2025-04-22 14:15:00 | 6505.00 | STOP_HIT | 0.50 | 3.10% |
| BUY | retest2 | 2025-04-23 11:30:00 | 6496.50 | 2025-04-25 10:15:00 | 6381.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-02 11:30:00 | 6315.00 | 2025-05-06 09:15:00 | 6476.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-05-02 13:15:00 | 6318.00 | 2025-05-06 09:15:00 | 6476.50 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-05-20 11:30:00 | 7228.50 | 2025-05-20 13:15:00 | 7106.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-20 12:30:00 | 7225.50 | 2025-05-20 13:15:00 | 7106.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-30 09:30:00 | 7616.50 | 2025-05-30 12:15:00 | 7480.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-06-09 10:15:00 | 7478.50 | 2025-06-13 09:15:00 | 7104.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 12:15:00 | 7472.50 | 2025-06-13 09:15:00 | 7098.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:00:00 | 7480.00 | 2025-06-13 09:15:00 | 7106.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:45:00 | 7472.50 | 2025-06-13 09:15:00 | 7098.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 10:45:00 | 7420.50 | 2025-06-16 09:15:00 | 7049.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 11:45:00 | 7435.00 | 2025-06-16 09:15:00 | 7063.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:15:00 | 7425.00 | 2025-06-16 09:15:00 | 7053.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:15:00 | 7478.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-06-09 12:15:00 | 7472.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-06-09 13:00:00 | 7480.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-06-09 13:45:00 | 7472.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-06-10 10:45:00 | 7420.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-06-11 11:45:00 | 7435.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2025-06-11 12:15:00 | 7425.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2025-07-03 13:30:00 | 6625.00 | 2025-07-03 14:15:00 | 6595.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-14 14:00:00 | 6844.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-15 09:15:00 | 6869.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-15 11:00:00 | 6844.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-15 13:00:00 | 6840.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-21 15:15:00 | 6675.00 | 2025-07-29 15:15:00 | 6570.00 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-07-24 10:00:00 | 6675.00 | 2025-07-29 15:15:00 | 6570.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-07-31 11:45:00 | 6615.00 | 2025-08-01 11:15:00 | 6561.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-01 10:15:00 | 6587.00 | 2025-08-01 11:15:00 | 6561.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-08 09:15:00 | 6401.00 | 2025-08-18 13:15:00 | 6276.50 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-08-21 10:15:00 | 6389.00 | 2025-08-21 15:15:00 | 6326.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-21 14:30:00 | 6399.00 | 2025-08-21 15:15:00 | 6326.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-25 11:30:00 | 6288.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-08-25 12:00:00 | 6279.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-25 13:45:00 | 6295.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-25 15:15:00 | 6222.00 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-03 14:45:00 | 6429.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-04 09:30:00 | 6460.50 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-09-05 11:30:00 | 6430.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-08 09:15:00 | 6435.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-10 09:30:00 | 6489.00 | 2025-09-11 11:15:00 | 6442.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-10 11:30:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-10 13:30:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-10 14:00:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-11 09:15:00 | 6511.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-09-26 11:15:00 | 6315.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2025-09-26 15:00:00 | 6306.50 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-09-29 09:15:00 | 6285.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-09-29 10:00:00 | 6299.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2025-10-01 12:15:00 | 6296.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-10-14 09:15:00 | 6220.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-14 10:15:00 | 6230.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-14 14:30:00 | 6206.50 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-15 09:30:00 | 6229.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-15 15:15:00 | 6210.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-16 15:15:00 | 6197.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-24 09:15:00 | 6089.00 | 2025-11-11 10:15:00 | 5784.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 12:30:00 | 6080.00 | 2025-11-11 10:15:00 | 5776.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 14:30:00 | 6082.50 | 2025-11-11 10:15:00 | 5778.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 09:45:00 | 6082.00 | 2025-11-11 10:15:00 | 5777.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 10:45:00 | 6078.00 | 2025-11-11 10:15:00 | 5774.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:00:00 | 6086.00 | 2025-11-11 10:15:00 | 5781.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:45:00 | 6085.00 | 2025-11-11 10:15:00 | 5780.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 13:15:00 | 6089.00 | 2025-11-11 10:15:00 | 5784.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:15:00 | 6070.00 | 2025-11-11 10:15:00 | 5766.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 6089.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-10-24 12:30:00 | 6080.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-10-24 14:30:00 | 6082.50 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-10-27 09:45:00 | 6082.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-10-28 10:45:00 | 6078.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-10-28 12:00:00 | 6086.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-10-28 12:45:00 | 6085.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-10-29 13:15:00 | 6089.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-10-29 14:15:00 | 6070.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-12-04 13:45:00 | 5799.00 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-04 14:30:00 | 5781.50 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-12-04 15:15:00 | 5787.00 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-12-11 09:15:00 | 5916.00 | 2025-12-17 10:15:00 | 5952.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-12-11 10:00:00 | 5899.50 | 2025-12-17 10:15:00 | 5952.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-12-22 11:15:00 | 5900.50 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-22 14:15:00 | 5910.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-22 15:15:00 | 5910.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-23 10:45:00 | 5910.50 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-24 10:30:00 | 5859.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-12-24 11:45:00 | 5855.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-24 13:15:00 | 5848.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-01-09 14:30:00 | 5918.50 | 2026-01-12 09:15:00 | 6197.00 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2026-01-09 15:00:00 | 5908.50 | 2026-01-12 09:15:00 | 6197.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-01-23 10:15:00 | 5911.00 | 2026-01-28 13:15:00 | 5914.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-02-13 09:15:00 | 6550.00 | 2026-02-23 09:15:00 | 6749.00 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2026-03-02 09:45:00 | 6789.50 | 2026-03-02 11:15:00 | 6736.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-03-02 10:30:00 | 6800.50 | 2026-03-02 11:15:00 | 6736.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-05 11:45:00 | 6641.00 | 2026-03-05 14:15:00 | 6760.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-03-17 11:30:00 | 7246.00 | 2026-03-19 13:15:00 | 7080.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-03-20 12:15:00 | 7066.00 | 2026-03-23 10:15:00 | 6712.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 7056.00 | 2026-03-23 10:15:00 | 6703.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 7066.00 | 2026-03-24 09:15:00 | 6962.50 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2026-03-20 13:45:00 | 7056.00 | 2026-03-24 09:15:00 | 6962.50 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2026-03-24 12:00:00 | 7060.00 | 2026-03-24 13:15:00 | 7415.00 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-04-01 10:15:00 | 6870.00 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2026-04-01 14:15:00 | 6947.00 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2026-04-01 15:15:00 | 6945.50 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-04-08 09:30:00 | 7229.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-08 14:30:00 | 7235.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-09 09:15:00 | 7243.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-09 10:30:00 | 7214.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-04-10 09:15:00 | 7336.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-04-13 10:00:00 | 7238.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-24 09:15:00 | 7271.50 | 2026-04-24 13:15:00 | 7199.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 7275.50 | 2026-04-27 11:15:00 | 7208.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-29 09:15:00 | 7262.00 | 2026-05-07 10:15:00 | 7988.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 10:00:00 | 7245.00 | 2026-05-07 10:15:00 | 7969.50 | TARGET_HIT | 1.00 | 10.00% |
