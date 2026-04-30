# MRF Ltd. (MRF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 129710.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 55.70
- **Avg P&L per closed trade:** 9.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 12:15:00 | 129200.00 | 135934.23 | 135954.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 127994.05 | 135309.95 | 135632.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 133901.00 | 133157.59 | 134345.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-03 09:15:00 | 130472.90 | 133199.90 | 134287.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-27 09:15:00 | 132210.05 | 130081.89 | 131801.10 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 135760.05 | 129550.67 | 129545.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 136147.25 | 129616.31 | 129578.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 132846.70 | 134074.84 | 132257.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-08 13:15:00 | 141273.66 | 134096.38 | 132303.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 141273.66 | 134096.38 | 132303.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-29 11:15:00 | 133800.05 | 136203.87 | 134262.93 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 129700.00 | 134807.81 | 134827.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 128999.00 | 134299.31 | 134565.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 125491.75 | 124768.67 | 127957.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 11:15:00 | 125159.70 | 124781.42 | 127932.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 09:15:00 | 129421.00 | 124907.71 | 127245.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 130650.30 | 128772.64 | 128771.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 132570.00 | 128875.53 | 128823.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 128314.00 | 129259.13 | 129038.18 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 122707.00 | 128784.91 | 128814.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 122535.80 | 128664.46 | 128754.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 108800.05 | 108711.45 | 112748.85 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 126475.00 | 114089.60 | 114034.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 126740.00 | 114215.47 | 114097.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 136715.00 | 136852.11 | 130534.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 09:15:00 | 138555.00 | 136849.81 | 130689.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 135020.00 | 137195.13 | 132996.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-24 11:15:00 | 136900.00 | 137033.44 | 133137.42 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 14:15:00 | 142500.00 | 145994.99 | 142586.36 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 148760.00 | 153076.03 | 153089.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 147910.00 | 151917.10 | 152440.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 143585.00 | 141906.17 | 145955.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 136315.00 | 144372.26 | 145951.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 14:15:00 | 138020.00 | 133927.84 | 137576.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-03 09:15:00 | 130472.90 | 2024-05-27 09:15:00 | 132210.05 | EXIT_EMA400 | -1737.15 |
| BUY | 2024-08-08 13:15:00 | 141273.66 | 2024-08-29 11:15:00 | 133800.05 | EXIT_EMA400 | -7473.61 |
| SELL | 2024-11-25 11:15:00 | 125159.70 | 2024-12-05 09:15:00 | 129421.00 | EXIT_EMA400 | -4261.30 |
| BUY | 2025-06-24 11:15:00 | 136900.00 | 2025-07-09 11:15:00 | 148187.75 | TARGET | 11287.75 |
| BUY | 2025-06-05 09:15:00 | 138555.00 | 2025-08-08 14:15:00 | 142500.00 | EXIT_EMA400 | 3945.00 |
| SELL | 2026-03-04 09:15:00 | 136315.00 | 2026-04-15 14:15:00 | 138020.00 | EXIT_EMA400 | -1705.00 |
