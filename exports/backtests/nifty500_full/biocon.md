# Biocon Ltd. (BIOCON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 359.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 1
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 3 / 8
- **Total realized P&L (per unit):** -12.07
- **Avg P&L per closed trade:** -1.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 235.05 | 260.08 | 260.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 234.50 | 259.11 | 259.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 238.00 | 235.16 | 243.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-28 09:15:00 | 232.20 | 235.18 | 242.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 241.50 | 235.43 | 241.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-01 11:15:00 | 242.00 | 235.55 | 241.50 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 13:15:00 | 247.30 | 243.87 | 243.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 09:15:00 | 248.35 | 243.98 | 243.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 12:15:00 | 266.10 | 267.47 | 258.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-01 09:15:00 | 269.60 | 265.91 | 259.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 265.65 | 272.00 | 264.50 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-13 10:15:00 | 270.60 | 271.99 | 264.53 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 269.80 | 271.84 | 264.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-14 15:15:00 | 271.70 | 271.75 | 264.84 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-02-29 09:15:00 | 268.35 | 274.46 | 268.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 250.20 | 266.69 | 266.77 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 269.00 | 266.43 | 266.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 269.35 | 266.48 | 266.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 267.50 | 268.40 | 267.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-18 12:15:00 | 272.25 | 268.18 | 267.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-18 14:15:00 | 266.70 | 268.21 | 267.45 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 332.70 | 353.53 | 353.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 329.20 | 353.08 | 353.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 335.95 | 334.21 | 341.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-18 10:15:00 | 326.15 | 335.43 | 341.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 338.70 | 333.86 | 339.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-26 10:15:00 | 342.05 | 333.94 | 339.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 373.50 | 344.39 | 344.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 380.00 | 346.08 | 345.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 350.30 | 354.04 | 349.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 10:15:00 | 356.90 | 353.67 | 350.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 350.80 | 353.66 | 350.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 349.45 | 353.62 | 350.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 14:15:00 | 336.70 | 362.83 | 362.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 328.10 | 362.22 | 362.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 337.00 | 336.70 | 345.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 324.65 | 340.72 | 345.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 334.25 | 332.65 | 338.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-24 09:15:00 | 327.95 | 332.82 | 338.85 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-05 13:15:00 | 335.65 | 328.82 | 335.35 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 14:15:00 | 352.90 | 336.61 | 336.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 09:15:00 | 354.60 | 336.93 | 336.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 379.30 | 380.66 | 367.53 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 356.45 | 362.64 | 362.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 355.15 | 362.46 | 362.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 355.75 | 355.03 | 358.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 10:15:00 | 348.45 | 354.84 | 357.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-16 09:15:00 | 358.10 | 354.30 | 357.34 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 375.40 | 359.28 | 359.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 380.40 | 362.29 | 360.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 388.75 | 391.03 | 380.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 399.40 | 388.66 | 382.56 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-06 14:15:00 | 385.00 | 391.54 | 386.47 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 371.00 | 383.34 | 383.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 369.95 | 383.21 | 383.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 374.90 | 374.36 | 377.74 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 390.45 | 379.75 | 379.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 396.75 | 382.44 | 381.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 384.05 | 384.91 | 382.66 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 14:15:00 | 369.20 | 381.04 | 381.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 362.85 | 380.10 | 380.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 364.50 | 363.45 | 370.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 09:15:00 | 358.50 | 363.33 | 370.04 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-28 09:15:00 | 232.20 | 2023-12-01 11:15:00 | 242.00 | EXIT_EMA400 | -9.80 |
| BUY | 2024-02-01 09:15:00 | 269.60 | 2024-02-06 11:15:00 | 300.19 | TARGET | 30.59 |
| BUY | 2024-02-13 10:15:00 | 270.60 | 2024-02-16 12:15:00 | 288.81 | TARGET | 18.21 |
| BUY | 2024-02-14 15:15:00 | 271.70 | 2024-02-19 09:15:00 | 292.27 | TARGET | 20.57 |
| BUY | 2024-04-18 12:15:00 | 272.25 | 2024-04-18 14:15:00 | 266.70 | EXIT_EMA400 | -5.55 |
| SELL | 2024-11-18 10:15:00 | 326.15 | 2024-11-26 10:15:00 | 342.05 | EXIT_EMA400 | -15.90 |
| BUY | 2024-12-18 10:15:00 | 356.90 | 2024-12-18 13:15:00 | 349.45 | EXIT_EMA400 | -7.45 |
| SELL | 2025-04-04 09:15:00 | 324.65 | 2025-05-05 13:15:00 | 335.65 | EXIT_EMA400 | -11.00 |
| SELL | 2025-04-24 09:15:00 | 327.95 | 2025-05-05 13:15:00 | 335.65 | EXIT_EMA400 | -7.70 |
| SELL | 2025-10-13 10:15:00 | 348.45 | 2025-10-16 09:15:00 | 358.10 | EXIT_EMA400 | -9.65 |
| BUY | 2025-12-19 09:15:00 | 399.40 | 2026-01-06 14:15:00 | 385.00 | EXIT_EMA400 | -14.40 |
