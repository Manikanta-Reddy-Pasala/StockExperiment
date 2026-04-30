# Polycab India Ltd. (POLYCAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 8110.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -750.98
- **Avg P&L per closed trade:** -150.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 11:15:00 | 3974.95 | 5223.19 | 5227.59 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 5089.00 | 4794.17 | 4793.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 5120.00 | 4833.47 | 4814.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 09:15:00 | 6699.85 | 6806.79 | 6367.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-28 10:15:00 | 6760.00 | 6806.32 | 6369.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 6445.55 | 6748.27 | 6432.97 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-10 09:15:00 | 6381.70 | 6730.62 | 6433.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 6399.00 | 6773.05 | 6773.86 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 12:15:00 | 6860.05 | 6774.93 | 6774.60 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 6659.50 | 6773.97 | 6774.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 6610.70 | 6767.32 | 6770.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 6733.00 | 6656.38 | 6707.27 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 7287.95 | 6752.53 | 6750.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 7363.35 | 6769.44 | 6759.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 7152.55 | 7190.10 | 7024.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 7268.15 | 7174.55 | 7039.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 7045.90 | 7205.19 | 7078.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 6481.10 | 6986.98 | 6988.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 6437.40 | 6905.91 | 6944.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 5379.85 | 5358.39 | 5774.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 4954.95 | 5357.14 | 5761.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 5529.00 | 5208.68 | 5460.87 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 5979.50 | 5576.90 | 5576.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 15:15:00 | 5981.00 | 5580.92 | 5578.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 5891.00 | 5954.12 | 5840.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 14:15:00 | 5999.50 | 5944.95 | 5842.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 7496.00 | 7607.55 | 7460.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 11:15:00 | 7430.50 | 7604.51 | 7460.43 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 7305.50 | 7393.94 | 7394.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 7271.00 | 7392.72 | 7393.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 7385.00 | 7351.00 | 7371.32 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 7653.00 | 7391.19 | 7390.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 7678.50 | 7454.51 | 7425.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 7536.50 | 7572.13 | 7499.04 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 7038.50 | 7443.23 | 7444.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 6992.00 | 7434.55 | 7439.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 7437.50 | 7247.04 | 7333.82 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 7796.00 | 7401.37 | 7400.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 7812.50 | 7405.46 | 7402.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 7956.50 | 7968.13 | 7748.25 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 6800.00 | 7594.54 | 7597.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 6769.50 | 7418.19 | 7501.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 7540.00 | 7346.74 | 7455.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 7461.00 | 7401.13 | 7472.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7461.00 | 7401.13 | 7472.76 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 7507.50 | 7402.19 | 7472.93 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 8220.00 | 7531.81 | 7531.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 8263.00 | 7539.09 | 7535.32 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-28 10:15:00 | 6760.00 | 2024-07-10 09:15:00 | 6381.70 | EXIT_EMA400 | -378.30 |
| BUY | 2024-12-30 09:15:00 | 7268.15 | 2025-01-06 10:15:00 | 7045.90 | EXIT_EMA400 | -222.25 |
| SELL | 2025-03-20 09:15:00 | 4954.95 | 2025-04-22 09:15:00 | 5529.00 | EXIT_EMA400 | -574.05 |
| BUY | 2025-06-20 14:15:00 | 5999.50 | 2025-06-25 09:15:00 | 6469.62 | TARGET | 470.12 |
| SELL | 2026-04-13 09:15:00 | 7461.00 | 2026-04-13 10:15:00 | 7507.50 | EXIT_EMA400 | -46.50 |
