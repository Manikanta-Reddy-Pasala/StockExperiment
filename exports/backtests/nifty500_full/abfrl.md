# Aditya Birla Fashion and Retail Ltd. (ABFRL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 64.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 48.01
- **Avg P&L per closed trade:** 8.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 14:15:00 | 215.45 | 218.32 | 218.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 15:15:00 | 215.05 | 218.29 | 218.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 220.30 | 217.51 | 217.88 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 231.65 | 218.26 | 218.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 235.00 | 220.77 | 219.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 226.70 | 226.88 | 223.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 09:15:00 | 229.35 | 226.92 | 223.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-20 13:15:00 | 220.75 | 228.10 | 224.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 13:15:00 | 222.00 | 232.35 | 232.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 220.30 | 232.23 | 232.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 245.15 | 217.25 | 223.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 229.80 | 225.93 | 226.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 229.80 | 225.93 | 226.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-15 10:15:00 | 230.30 | 225.97 | 226.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 12:15:00 | 231.30 | 227.17 | 227.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 13:15:00 | 232.15 | 227.22 | 227.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 268.70 | 269.48 | 255.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 10:15:00 | 291.60 | 269.69 | 256.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 311.35 | 316.52 | 300.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-23 13:15:00 | 315.00 | 316.51 | 301.05 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-13 15:15:00 | 311.30 | 323.19 | 311.94 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 304.40 | 323.90 | 323.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 301.20 | 322.67 | 323.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 306.00 | 305.43 | 312.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 298.80 | 308.48 | 311.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 285.00 | 277.51 | 286.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-03 10:15:00 | 290.25 | 277.64 | 286.68 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 278.15 | 262.39 | 262.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 281.15 | 263.98 | 263.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 89.95 | 265.69 | 264.28 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 90.45 | 262.23 | 262.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 90.00 | 260.51 | 261.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 79.92 | 77.73 | 95.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-24 14:15:00 | 75.39 | 80.63 | 83.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 11:15:00 | 73.63 | 69.90 | 73.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-14 09:15:00 | 229.35 | 2023-12-20 13:15:00 | 220.75 | EXIT_EMA400 | -8.60 |
| SELL | 2024-04-15 09:15:00 | 229.80 | 2024-04-15 10:15:00 | 230.30 | EXIT_EMA400 | -0.50 |
| BUY | 2024-06-05 10:15:00 | 291.60 | 2024-08-13 15:15:00 | 311.30 | EXIT_EMA400 | 19.70 |
| BUY | 2024-07-23 13:15:00 | 315.00 | 2024-08-13 15:15:00 | 311.30 | EXIT_EMA400 | -3.70 |
| SELL | 2024-12-13 09:15:00 | 298.80 | 2025-01-13 14:15:00 | 259.45 | TARGET | 39.35 |
| SELL | 2025-11-24 14:15:00 | 75.39 | 2026-02-09 11:15:00 | 73.63 | EXIT_EMA400 | 1.76 |
