# CRISIL Ltd. (CRISIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4266.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -685.30
- **Avg P&L per closed trade:** -171.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 14:15:00 | 5122.65 | 5437.16 | 5437.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 5092.55 | 5433.73 | 5435.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 14:15:00 | 4369.80 | 4367.65 | 4624.93 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 5019.10 | 4696.81 | 4695.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 5032.70 | 4703.38 | 4698.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 5779.50 | 5786.03 | 5551.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 12:15:00 | 5803.50 | 5786.26 | 5553.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 5538.50 | 5777.56 | 5559.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 5302.50 | 5434.01 | 5434.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5418.35 | 5425.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 4786.40 | 4783.88 | 4955.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-07 15:15:00 | 4716.50 | 4846.66 | 4937.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-06 11:15:00 | 4546.40 | 4415.49 | 4543.78 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 4653.60 | 4610.41 | 4610.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 4689.30 | 4612.46 | 4611.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-12 12:15:00 | 4666.80 | 4634.46 | 4623.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 4666.80 | 4634.46 | 4623.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-13 09:15:00 | 4365.80 | 4632.78 | 4622.91 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 4512.90 | 4613.57 | 4613.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4465.80 | 4601.10 | 4607.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 4064.10 | 4063.83 | 4238.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 4006.10 | 4064.08 | 4236.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4201.40 | 4074.76 | 4223.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 12:15:00 | 4295.50 | 4076.96 | 4223.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-24 12:15:00 | 5803.50 | 2025-07-25 14:15:00 | 5538.50 | EXIT_EMA400 | -265.00 |
| SELL | 2025-11-07 15:15:00 | 4716.50 | 2026-01-06 11:15:00 | 4546.40 | EXIT_EMA400 | 170.10 |
| BUY | 2026-02-12 12:15:00 | 4666.80 | 2026-02-13 09:15:00 | 4365.80 | EXIT_EMA400 | -301.00 |
| SELL | 2026-04-13 09:15:00 | 4006.10 | 2026-04-17 12:15:00 | 4295.50 | EXIT_EMA400 | -289.40 |
