# Amber Enterprises India Ltd. (AMBER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 8024.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 285.35
- **Avg P&L per closed trade:** 47.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 11:15:00 | 3279.90 | 3615.58 | 3617.00 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 12:15:00 | 3767.95 | 3617.23 | 3616.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 11:15:00 | 3816.15 | 3645.11 | 3631.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 3640.00 | 3675.70 | 3650.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 13:15:00 | 3640.70 | 3673.92 | 3649.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 3640.70 | 3673.92 | 3649.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-15 14:15:00 | 3602.00 | 3673.20 | 3649.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 5517.80 | 6476.76 | 6476.88 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 15:15:00 | 7248.50 | 6367.86 | 6367.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 7298.00 | 6542.00 | 6462.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 6625.00 | 6649.52 | 6532.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 6865.00 | 6571.04 | 6506.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-24 09:15:00 | 6453.50 | 6621.45 | 6547.85 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 15:15:00 | 6098.00 | 6488.92 | 6490.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 6012.00 | 6484.18 | 6488.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 6350.00 | 6345.04 | 6407.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-19 09:15:00 | 6236.50 | 6351.70 | 6406.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 6236.50 | 6351.70 | 6406.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-20 09:15:00 | 6502.50 | 6348.69 | 6403.41 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 6575.00 | 6433.19 | 6432.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 6626.50 | 6447.16 | 6440.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6449.00 | 6459.71 | 6446.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 6561.50 | 6461.26 | 6448.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 6561.50 | 6461.26 | 6448.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-23 11:15:00 | 6709.50 | 6464.96 | 6450.32 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-11 09:15:00 | 7053.00 | 7490.01 | 7204.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 7327.50 | 7824.55 | 7825.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7708.55 | 7763.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6219.00 | 6156.00 | 6543.06 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 7820.00 | 6759.91 | 6759.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7828.00 | 6800.79 | 6780.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 7340.00 | 7371.15 | 7132.29 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 6761.50 | 7003.30 | 7003.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 6567.50 | 6991.87 | 6997.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 7048.00 | 6812.68 | 6897.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 6782.50 | 6818.54 | 6896.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-09 14:15:00 | 6898.00 | 6819.33 | 6896.68 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 7874.00 | 6965.17 | 6962.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 7976.00 | 7009.32 | 6985.35 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-15 13:15:00 | 3640.70 | 2024-04-15 14:15:00 | 3602.00 | EXIT_EMA400 | -38.70 |
| BUY | 2025-04-15 09:15:00 | 6865.00 | 2025-04-24 09:15:00 | 6453.50 | EXIT_EMA400 | -411.50 |
| SELL | 2025-05-19 09:15:00 | 6236.50 | 2025-05-20 09:15:00 | 6502.50 | EXIT_EMA400 | -266.00 |
| BUY | 2025-06-23 09:15:00 | 6561.50 | 2025-06-25 11:15:00 | 6901.01 | TARGET | 339.51 |
| BUY | 2025-06-23 11:15:00 | 6709.50 | 2025-07-08 09:15:00 | 7487.04 | TARGET | 777.54 |
| SELL | 2026-04-09 13:15:00 | 6782.50 | 2026-04-09 14:15:00 | 6898.00 | EXIT_EMA400 | -115.50 |
