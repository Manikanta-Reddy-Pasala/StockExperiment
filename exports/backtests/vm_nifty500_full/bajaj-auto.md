# Bajaj Auto Ltd. (BAJAJ-AUTO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 9994.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 2
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 805.70
- **Avg P&L per closed trade:** 100.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 4816.20 | 4715.59 | 4715.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 4843.10 | 4719.96 | 4717.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 15:15:00 | 4906.05 | 4911.34 | 4836.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-05 09:15:00 | 4954.95 | 4911.78 | 4836.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 9416.00 | 9500.12 | 9294.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-18 12:15:00 | 9496.15 | 9500.08 | 9295.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 9392.30 | 9499.09 | 9305.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-23 10:15:00 | 9470.10 | 9492.10 | 9309.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-23 12:15:00 | 9306.00 | 9490.05 | 9310.35 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 9400.80 | 10773.45 | 10779.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10073.77 | 10352.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.60 | 9054.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 10:15:00 | 8573.00 | 8777.57 | 8986.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 8113.50 | 7882.53 | 8128.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 10:15:00 | 8178.00 | 7885.47 | 8129.13 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 8813.00 | 8176.93 | 8169.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 8464.00 | 8493.69 | 8379.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 8328.00 | 8498.91 | 8397.96 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.90 | 8364.00 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 8405.50 | 8364.19 | 8364.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 8445.00 | 8365.00 | 8364.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8303.00 | 8363.81 | 8363.94 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 8394.00 | 8364.29 | 8364.17 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 8285.50 | 8363.45 | 8363.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 8062.50 | 8359.72 | 8361.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.48 | 8291.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-08 13:15:00 | 8203.50 | 8241.65 | 8289.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8239.11 | 8286.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-11 15:15:00 | 8235.00 | 8239.07 | 8286.40 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-12 09:15:00 | 8293.50 | 8239.62 | 8286.44 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8827.00 | 8322.51 | 8321.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.45 | 8403.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.58 | 8716.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 12:15:00 | 8948.50 | 8831.05 | 8733.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 8885.00 | 8972.21 | 8857.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-04 09:15:00 | 8798.00 | 8963.42 | 8859.35 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9427.06 | 9431.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 9769.50 | 9371.94 | 9371.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 9817.00 | 9399.20 | 9385.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9441.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9441.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-30 12:15:00 | 9835.00 | 9496.88 | 9444.80 | Buy entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-05 09:15:00 | 4954.95 | 2023-10-19 09:15:00 | 5309.33 | TARGET | 354.38 |
| BUY | 2024-07-18 12:15:00 | 9496.15 | 2024-07-23 12:15:00 | 9306.00 | EXIT_EMA400 | -190.15 |
| BUY | 2024-07-23 10:15:00 | 9470.10 | 2024-07-23 12:15:00 | 9306.00 | EXIT_EMA400 | -164.10 |
| SELL | 2025-02-14 10:15:00 | 8573.00 | 2025-03-04 13:15:00 | 7332.43 | TARGET | 1240.57 |
| BUY | 2025-06-13 14:15:00 | 8464.00 | 2025-06-20 09:15:00 | 8328.00 | EXIT_EMA400 | -136.00 |
| SELL | 2025-08-08 13:15:00 | 8203.50 | 2025-08-12 09:15:00 | 8293.50 | EXIT_EMA400 | -90.00 |
| SELL | 2025-08-11 15:15:00 | 8235.00 | 2025-08-12 09:15:00 | 8293.50 | EXIT_EMA400 | -58.50 |
| BUY | 2025-10-10 12:15:00 | 8948.50 | 2025-11-04 09:15:00 | 8798.00 | EXIT_EMA400 | -150.50 |
