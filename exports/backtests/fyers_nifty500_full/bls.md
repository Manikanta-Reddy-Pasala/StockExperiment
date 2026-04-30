# BLS International Services Ltd. (BLS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 277.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 70.54
- **Avg P&L per closed trade:** 11.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 363.25 | 385.27 | 385.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 361.00 | 384.81 | 385.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 383.05 | 382.89 | 384.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 12:15:00 | 373.40 | 382.10 | 383.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-23 11:15:00 | 385.05 | 379.40 | 382.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 412.40 | 383.67 | 383.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 417.45 | 384.01 | 383.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 386.00 | 393.09 | 388.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-03 09:15:00 | 410.65 | 391.27 | 389.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 441.45 | 463.29 | 440.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 438.30 | 463.04 | 440.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 395.40 | 436.75 | 436.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 388.90 | 435.11 | 436.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 366.50 | 365.23 | 388.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 350.75 | 381.43 | 391.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 385.65 | 375.70 | 386.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 11:15:00 | 386.90 | 375.81 | 386.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 419.40 | 383.69 | 383.54 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 364.90 | 386.27 | 386.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 362.10 | 385.81 | 386.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-30 09:15:00 | 370.55 | 377.36 | 381.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 377.36 | 381.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-01 09:15:00 | 363.75 | 376.80 | 380.75 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-07 09:15:00 | 382.30 | 373.76 | 378.60 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 397.50 | 379.60 | 379.52 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 373.00 | 379.92 | 379.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 368.85 | 379.67 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 355.25 | 353.95 | 363.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-08 11:15:00 | 349.00 | 353.88 | 362.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-12 12:15:00 | 337.50 | 323.97 | 337.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-21 12:15:00 | 373.40 | 2024-10-23 09:15:00 | 343.07 | TARGET | 30.33 |
| BUY | 2024-12-03 09:15:00 | 410.65 | 2024-12-10 09:15:00 | 475.38 | TARGET | 64.73 |
| SELL | 2025-04-07 09:15:00 | 350.75 | 2025-04-15 11:15:00 | 386.90 | EXIT_EMA400 | -36.15 |
| SELL | 2025-06-30 09:15:00 | 370.55 | 2025-07-07 09:15:00 | 382.30 | EXIT_EMA400 | -11.75 |
| SELL | 2025-07-01 09:15:00 | 363.75 | 2025-07-07 09:15:00 | 382.30 | EXIT_EMA400 | -18.55 |
| SELL | 2025-10-08 11:15:00 | 349.00 | 2025-10-13 09:15:00 | 307.08 | TARGET | 41.92 |
