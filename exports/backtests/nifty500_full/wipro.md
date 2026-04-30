# Wipro Ltd. (WIPRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 200.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -59.28
- **Avg P&L per closed trade:** -9.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 197.35 | 206.48 | 206.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 196.05 | 206.28 | 206.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 196.38 | 196.31 | 199.94 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 209.98 | 201.36 | 201.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 210.65 | 201.62 | 201.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 252.23 | 253.87 | 242.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-06 15:15:00 | 256.58 | 253.86 | 243.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 12:15:00 | 225.00 | 242.68 | 242.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 223.93 | 242.50 | 242.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 233.12 | 232.36 | 235.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-27 09:15:00 | 227.85 | 232.40 | 235.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 242.05 | 228.63 | 232.41 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 248.38 | 235.04 | 235.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 253.18 | 238.18 | 236.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 255.40 | 261.20 | 251.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 09:15:00 | 263.23 | 259.10 | 251.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-02 13:15:00 | 253.07 | 259.47 | 253.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 15:15:00 | 285.20 | 299.05 | 299.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 283.25 | 298.65 | 298.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 250.78 | 249.04 | 261.28 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 265.01 | 258.50 | 258.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 266.11 | 258.92 | 258.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 13:15:00 | 262.45 | 261.40 | 260.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 261.30 | 261.43 | 260.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-17 15:15:00 | 259.00 | 261.41 | 260.47 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 250.20 | 259.81 | 259.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 250.05 | 253.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 09:15:00 | 248.27 | 250.12 | 253.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-25 09:15:00 | 254.75 | 250.11 | 253.46 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.33 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.79 | 257.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.62 | 254.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 250.08 | 252.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.79 | 212.63 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-06 15:15:00 | 256.58 | 2024-03-20 09:15:00 | 245.27 | EXIT_EMA400 | -11.30 |
| SELL | 2024-05-27 09:15:00 | 227.85 | 2024-06-07 09:15:00 | 242.05 | EXIT_EMA400 | -14.20 |
| BUY | 2024-07-29 09:15:00 | 263.23 | 2024-08-02 13:15:00 | 253.07 | EXIT_EMA400 | -10.15 |
| BUY | 2025-07-16 13:15:00 | 262.45 | 2025-07-17 15:15:00 | 259.00 | EXIT_EMA400 | -3.45 |
| SELL | 2025-08-22 09:15:00 | 248.27 | 2025-08-25 09:15:00 | 254.75 | EXIT_EMA400 | -6.48 |
| BUY | 2026-01-12 12:15:00 | 263.70 | 2026-01-19 09:15:00 | 250.00 | EXIT_EMA400 | -13.70 |
