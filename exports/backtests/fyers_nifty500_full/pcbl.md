# PCBL Chemical Ltd. (PCBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 288.90
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
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -62.08
- **Avg P&L per closed trade:** -10.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 11:15:00 | 396.40 | 447.33 | 447.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 14:15:00 | 391.70 | 445.78 | 446.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 433.90 | 427.44 | 436.05 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 485.75 | 441.25 | 441.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 490.50 | 441.74 | 441.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 457.35 | 457.47 | 450.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-26 10:15:00 | 462.95 | 457.60 | 450.98 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-30 12:15:00 | 449.25 | 457.82 | 451.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 386.30 | 447.54 | 447.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 360.45 | 445.58 | 446.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 394.05 | 391.84 | 411.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 380.40 | 391.73 | 411.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-04 09:15:00 | 411.25 | 392.02 | 411.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 428.35 | 398.97 | 398.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 432.50 | 399.30 | 399.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 381.20 | 403.27 | 401.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 399.00 | 402.52 | 400.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 399.00 | 402.52 | 400.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-08 14:15:00 | 399.55 | 402.53 | 400.85 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 359.85 | 401.62 | 401.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 354.00 | 390.96 | 396.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 391.75 | 386.35 | 392.66 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 417.50 | 395.98 | 395.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 429.50 | 396.75 | 396.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 401.30 | 402.30 | 399.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 10:15:00 | 408.40 | 398.26 | 397.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 408.40 | 398.26 | 397.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-25 14:15:00 | 410.35 | 398.60 | 398.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-07 15:15:00 | 401.20 | 404.68 | 401.80 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 388.00 | 404.06 | 404.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 383.50 | 403.32 | 403.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 394.75 | 394.74 | 398.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-20 13:15:00 | 389.40 | 394.54 | 398.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 396.00 | 385.92 | 391.65 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-26 10:15:00 | 462.95 | 2024-12-30 12:15:00 | 449.25 | EXIT_EMA400 | -13.70 |
| SELL | 2025-02-03 09:15:00 | 380.40 | 2025-02-04 09:15:00 | 411.25 | EXIT_EMA400 | -30.85 |
| BUY | 2025-04-08 09:15:00 | 399.00 | 2025-04-08 11:15:00 | 404.42 | TARGET | 5.42 |
| BUY | 2025-06-25 10:15:00 | 408.40 | 2025-07-07 15:15:00 | 401.20 | EXIT_EMA400 | -7.20 |
| BUY | 2025-06-25 14:15:00 | 410.35 | 2025-07-07 15:15:00 | 401.20 | EXIT_EMA400 | -9.15 |
| SELL | 2025-08-20 13:15:00 | 389.40 | 2025-09-10 09:15:00 | 396.00 | EXIT_EMA400 | -6.60 |
