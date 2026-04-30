# BLS International Services Ltd. (BLS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 278.31
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 6.66
- **Avg P&L per closed trade:** 0.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 321.95 | 348.13 | 348.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 318.40 | 347.57 | 347.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 09:15:00 | 344.50 | 341.96 | 344.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-12 13:15:00 | 336.60 | 343.27 | 345.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-29 10:15:00 | 341.90 | 338.16 | 341.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 361.75 | 332.91 | 332.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 368.55 | 338.56 | 335.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 351.75 | 358.93 | 349.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 10:15:00 | 363.85 | 352.67 | 348.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 361.40 | 355.08 | 350.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 15:15:00 | 348.00 | 354.87 | 350.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 364.70 | 386.41 | 386.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 363.30 | 386.18 | 386.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 383.60 | 382.91 | 384.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 12:15:00 | 373.40 | 382.11 | 384.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-23 11:15:00 | 385.05 | 379.41 | 382.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 417.50 | 384.22 | 384.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 420.90 | 385.86 | 385.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 385.75 | 392.77 | 388.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-03 09:15:00 | 410.65 | 391.12 | 389.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 441.45 | 463.26 | 441.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 438.30 | 463.01 | 440.99 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 393.75 | 436.68 | 436.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 389.15 | 435.46 | 436.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 366.50 | 365.32 | 388.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 350.75 | 381.46 | 391.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 385.70 | 375.73 | 386.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 12:15:00 | 387.20 | 375.95 | 386.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 419.15 | 383.67 | 383.53 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 364.90 | 386.28 | 386.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 362.10 | 385.82 | 386.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 378.75 | 377.43 | 381.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-30 09:15:00 | 370.55 | 377.38 | 381.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 377.38 | 381.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-01 09:15:00 | 363.75 | 376.83 | 380.76 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-07 09:15:00 | 382.30 | 373.78 | 378.60 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 397.50 | 379.61 | 379.53 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 373.00 | 379.92 | 379.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 368.85 | 379.67 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 355.05 | 353.94 | 363.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-08 11:15:00 | 349.10 | 353.88 | 362.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-12 12:15:00 | 337.50 | 323.98 | 337.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-12 13:15:00 | 336.60 | 2024-04-29 10:15:00 | 341.90 | EXIT_EMA400 | -5.30 |
| BUY | 2024-07-29 10:15:00 | 363.85 | 2024-08-05 15:15:00 | 348.00 | EXIT_EMA400 | -15.85 |
| SELL | 2024-10-21 12:15:00 | 373.40 | 2024-10-23 11:15:00 | 385.05 | EXIT_EMA400 | -11.65 |
| BUY | 2024-12-03 09:15:00 | 410.65 | 2024-12-10 09:15:00 | 475.25 | TARGET | 64.60 |
| SELL | 2025-04-07 09:15:00 | 350.75 | 2025-04-15 12:15:00 | 387.20 | EXIT_EMA400 | -36.45 |
| SELL | 2025-06-30 09:15:00 | 370.55 | 2025-07-07 09:15:00 | 382.30 | EXIT_EMA400 | -11.75 |
| SELL | 2025-07-01 09:15:00 | 363.75 | 2025-07-07 09:15:00 | 382.30 | EXIT_EMA400 | -18.55 |
| SELL | 2025-10-08 11:15:00 | 349.10 | 2025-10-13 09:15:00 | 307.49 | TARGET | 41.61 |
