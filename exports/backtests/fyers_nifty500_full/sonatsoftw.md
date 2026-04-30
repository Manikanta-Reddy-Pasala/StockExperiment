# Sonata Software Ltd. (SONATSOFTW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 255.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -52.17
- **Avg P&L per closed trade:** -8.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 577.00 | 633.04 | 633.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 570.35 | 630.01 | 631.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 12:15:00 | 625.00 | 624.63 | 628.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-18 09:15:00 | 605.65 | 624.40 | 628.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 617.05 | 612.76 | 620.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 09:15:00 | 599.70 | 612.63 | 620.14 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-06 12:15:00 | 619.25 | 611.87 | 619.13 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 670.00 | 614.16 | 613.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 673.05 | 617.96 | 615.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 633.80 | 640.69 | 629.69 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 596.20 | 623.14 | 623.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 580.80 | 622.46 | 622.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 11:15:00 | 368.20 | 349.18 | 396.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 09:15:00 | 340.20 | 349.44 | 395.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-30 10:15:00 | 397.35 | 351.78 | 392.22 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 423.35 | 401.36 | 401.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 425.95 | 403.58 | 402.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 404.85 | 406.06 | 403.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 10:15:00 | 414.05 | 404.37 | 403.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 404.80 | 406.44 | 404.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-04 10:15:00 | 410.15 | 406.43 | 404.71 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 414.80 | 422.38 | 415.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 356.95 | 410.35 | 410.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 354.00 | 408.25 | 409.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 384.20 | 380.61 | 391.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 374.70 | 380.83 | 391.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-12 09:15:00 | 385.20 | 371.12 | 381.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-18 09:15:00 | 605.65 | 2024-11-06 12:15:00 | 619.25 | EXIT_EMA400 | -13.60 |
| SELL | 2024-11-04 09:15:00 | 599.70 | 2024-11-06 12:15:00 | 619.25 | EXIT_EMA400 | -19.55 |
| SELL | 2025-04-25 09:15:00 | 340.20 | 2025-04-30 10:15:00 | 397.35 | EXIT_EMA400 | -57.15 |
| BUY | 2025-07-04 10:15:00 | 410.15 | 2025-07-08 09:15:00 | 426.46 | TARGET | 16.31 |
| BUY | 2025-06-25 10:15:00 | 414.05 | 2025-07-08 12:15:00 | 446.37 | TARGET | 32.32 |
| SELL | 2025-08-26 09:15:00 | 374.70 | 2025-09-12 09:15:00 | 385.20 | EXIT_EMA400 | -10.50 |
