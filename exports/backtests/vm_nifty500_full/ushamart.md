# Usha Martin Ltd. (USHAMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4997 bars)
- **Last close:** 451.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 2 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -139.05
- **Avg P&L per closed trade:** -17.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 266.50 | 325.90 | 326.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 10:15:00 | 257.60 | 325.22 | 325.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 314.15 | 311.23 | 317.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-08 14:15:00 | 310.80 | 311.30 | 317.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-13 09:15:00 | 319.70 | 311.14 | 316.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 13:15:00 | 332.90 | 320.11 | 320.09 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 15:15:00 | 308.95 | 320.12 | 320.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 09:15:00 | 306.05 | 319.98 | 320.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 316.75 | 314.03 | 316.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-20 12:15:00 | 308.05 | 313.96 | 316.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-01-04 12:15:00 | 313.80 | 307.39 | 312.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 11:15:00 | 355.30 | 314.32 | 314.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 14:15:00 | 357.50 | 315.55 | 314.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 331.50 | 335.77 | 327.04 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 14:15:00 | 308.30 | 322.90 | 322.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 304.30 | 320.57 | 321.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 307.50 | 305.95 | 312.90 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 13:15:00 | 344.75 | 317.73 | 317.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 14:15:00 | 348.00 | 324.24 | 321.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 344.45 | 349.04 | 337.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-22 12:15:00 | 358.00 | 345.86 | 338.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 348.95 | 351.64 | 343.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 314.95 | 351.27 | 343.77 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 335.95 | 368.13 | 368.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 331.00 | 365.84 | 367.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 350.85 | 345.85 | 353.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 09:15:00 | 342.65 | 351.20 | 354.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 350.30 | 350.48 | 354.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-25 09:15:00 | 354.70 | 350.53 | 353.84 | Close above EMA400 |

### Cycle 8 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 422.50 | 355.46 | 355.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 10:15:00 | 428.50 | 357.49 | 356.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 10:15:00 | 383.90 | 384.06 | 372.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-28 09:15:00 | 397.70 | 384.20 | 372.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-12 14:15:00 | 383.50 | 397.83 | 384.69 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 372.05 | 385.53 | 385.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 12:15:00 | 369.85 | 385.38 | 385.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 315.00 | 314.59 | 333.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 10:15:00 | 310.85 | 317.75 | 331.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 11:15:00 | 330.10 | 317.32 | 329.71 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 346.30 | 316.49 | 316.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 351.05 | 316.83 | 316.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 360.60 | 362.78 | 347.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 14:15:00 | 366.00 | 362.77 | 347.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 352.10 | 365.99 | 352.97 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 421.40 | 438.70 | 438.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 414.30 | 437.88 | 438.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 431.95 | 426.32 | 431.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 413.60 | 426.65 | 431.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 435.55 | 420.23 | 426.39 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 448.80 | 420.58 | 420.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 11:15:00 | 453.55 | 426.19 | 423.48 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-08 14:15:00 | 310.80 | 2023-11-13 09:15:00 | 319.70 | EXIT_EMA400 | -8.90 |
| SELL | 2023-12-20 12:15:00 | 308.05 | 2024-01-04 12:15:00 | 313.80 | EXIT_EMA400 | -5.75 |
| BUY | 2024-05-22 12:15:00 | 358.00 | 2024-06-04 10:15:00 | 314.95 | EXIT_EMA400 | -43.05 |
| SELL | 2024-09-19 09:15:00 | 342.65 | 2024-09-25 09:15:00 | 354.70 | EXIT_EMA400 | -12.05 |
| BUY | 2024-10-28 09:15:00 | 397.70 | 2024-11-12 14:15:00 | 383.50 | EXIT_EMA400 | -14.20 |
| SELL | 2025-03-13 10:15:00 | 310.85 | 2025-03-20 11:15:00 | 330.10 | EXIT_EMA400 | -19.25 |
| BUY | 2025-07-28 14:15:00 | 366.00 | 2025-08-07 09:15:00 | 352.10 | EXIT_EMA400 | -13.90 |
| SELL | 2026-02-13 09:15:00 | 413.60 | 2026-02-25 09:15:00 | 435.55 | EXIT_EMA400 | -21.95 |
