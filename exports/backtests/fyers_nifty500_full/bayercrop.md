# Bayer Cropscience Ltd. (BAYERCROP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4798.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 210.46
- **Avg P&L per closed trade:** 70.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 5650.00 | 6443.97 | 6444.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 12:15:00 | 5638.40 | 6435.95 | 6440.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 6119.00 | 6110.67 | 6248.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 14:15:00 | 6031.45 | 6107.84 | 6230.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-10 15:15:00 | 6291.00 | 6107.08 | 6221.47 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 5716.00 | 4939.39 | 4936.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 13:15:00 | 5745.50 | 4947.41 | 4940.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 5994.50 | 6198.11 | 5943.36 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 5273.00 | 5793.40 | 5795.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 5266.00 | 5788.16 | 5792.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 15:15:00 | 5110.00 | 5108.62 | 5281.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 09:15:00 | 5088.90 | 5108.42 | 5280.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4519.60 | 4441.85 | 4524.54 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 10:15:00 | 4561.80 | 4443.05 | 4524.73 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 15:15:00 | 4779.80 | 4584.44 | 4583.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 12:15:00 | 4800.10 | 4591.68 | 4587.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4463.00 | 4594.67 | 4594.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 4453.80 | 4593.27 | 4594.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-23 10:15:00 | 4490.00 | 4562.88 | 4576.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4535.00 | 4559.14 | 4574.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-24 13:15:00 | 4594.00 | 4558.61 | 4574.02 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 4777.30 | 4584.57 | 4583.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4590.77 | 4586.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 4698.80 | 4727.93 | 4670.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 11:15:00 | 4759.50 | 4717.26 | 4672.17 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-06 14:15:00 | 6031.45 | 2024-12-10 15:15:00 | 6291.00 | EXIT_EMA400 | -259.55 |
| SELL | 2025-10-17 09:15:00 | 5088.90 | 2025-11-11 13:15:00 | 4514.89 | TARGET | 574.01 |
| SELL | 2026-03-23 10:15:00 | 4490.00 | 2026-03-24 13:15:00 | 4594.00 | EXIT_EMA400 | -104.00 |
