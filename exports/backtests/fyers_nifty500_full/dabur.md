# Dabur India Ltd. (DABUR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 442.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 2
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -41.18
- **Avg P&L per closed trade:** -6.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 13:15:00 | 567.25 | 627.20 | 627.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 565.15 | 622.62 | 625.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 518.20 | 514.82 | 532.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 505.45 | 515.19 | 532.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 523.30 | 514.32 | 529.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-09 11:15:00 | 521.00 | 514.39 | 529.85 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-17 10:15:00 | 530.05 | 515.37 | 527.51 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 512.80 | 485.03 | 484.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 514.00 | 485.32 | 485.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 513.00 | 513.82 | 504.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 09:15:00 | 515.45 | 513.83 | 504.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-11 09:15:00 | 502.25 | 513.55 | 504.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 491.40 | 515.91 | 516.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 489.10 | 514.79 | 515.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 505.60 | 505.15 | 509.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 09:15:00 | 492.15 | 505.35 | 508.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 503.85 | 503.82 | 507.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-04 10:15:00 | 508.50 | 503.87 | 507.78 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 526.85 | 510.74 | 510.70 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 503.00 | 511.78 | 511.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 502.20 | 511.55 | 511.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 505.35 | 500.54 | 504.98 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 522.30 | 508.09 | 508.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 13:15:00 | 524.00 | 508.39 | 508.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 11:15:00 | 509.70 | 510.49 | 509.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-21 12:15:00 | 513.20 | 510.29 | 509.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-28 09:15:00 | 508.20 | 512.49 | 510.64 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 502.30 | 509.27 | 509.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 499.85 | 509.10 | 509.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 510.70 | 508.56 | 508.90 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 518.30 | 509.25 | 509.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 524.50 | 512.23 | 511.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 510.00 | 512.69 | 511.30 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 485.70 | 509.99 | 510.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 480.80 | 508.77 | 509.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 11:15:00 | 444.80 | 443.64 | 463.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-20 15:15:00 | 440.00 | 443.60 | 463.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 460.35 | 444.99 | 462.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-22 15:15:00 | 459.00 | 445.13 | 462.49 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 455.50 | 446.10 | 462.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-24 10:15:00 | 451.85 | 446.16 | 462.25 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 09:15:00 | 505.45 | 2025-01-17 10:15:00 | 530.05 | EXIT_EMA400 | -24.60 |
| SELL | 2025-01-09 11:15:00 | 521.00 | 2025-01-17 10:15:00 | 530.05 | EXIT_EMA400 | -9.05 |
| BUY | 2025-08-07 09:15:00 | 515.45 | 2025-08-11 09:15:00 | 502.25 | EXIT_EMA400 | -13.20 |
| SELL | 2025-10-31 09:15:00 | 492.15 | 2025-11-04 10:15:00 | 508.50 | EXIT_EMA400 | -16.35 |
| BUY | 2026-01-21 12:15:00 | 513.20 | 2026-01-22 09:15:00 | 524.75 | TARGET | 11.55 |
| SELL | 2026-04-22 15:15:00 | 459.00 | 2026-04-24 13:15:00 | 448.53 | TARGET | 10.47 |
