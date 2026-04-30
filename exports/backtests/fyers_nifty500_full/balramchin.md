# Balrampur Chini Mills Ltd. (BALRAMCHIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 522.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -66.29
- **Avg P&L per closed trade:** -16.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 559.20 | 579.48 | 579.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 552.40 | 577.98 | 578.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 578.00 | 577.15 | 578.36 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 609.00 | 579.61 | 579.51 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 538.00 | 580.12 | 580.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 529.95 | 579.62 | 579.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 13:15:00 | 470.25 | 468.44 | 496.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-25 12:15:00 | 457.35 | 469.04 | 493.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-12 14:15:00 | 489.15 | 460.92 | 481.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 545.40 | 493.79 | 493.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 546.60 | 494.32 | 493.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 545.15 | 545.90 | 528.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 562.15 | 546.34 | 529.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 539.05 | 547.35 | 530.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 11:15:00 | 528.05 | 547.01 | 530.93 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 550.00 | 581.55 | 581.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 547.20 | 579.92 | 580.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 12:15:00 | 568.65 | 577.24 | 578.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 577.70 | 570.81 | 575.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 454.75 | 444.83 | 444.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 465.30 | 445.63 | 445.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 12:15:00 | 469.30 | 470.94 | 460.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 09:15:00 | 482.85 | 471.01 | 460.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-24 09:15:00 | 452.00 | 472.74 | 463.10 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-25 12:15:00 | 457.35 | 2025-03-12 14:15:00 | 489.15 | EXIT_EMA400 | -31.80 |
| BUY | 2025-05-07 10:15:00 | 562.15 | 2025-05-09 11:15:00 | 528.05 | EXIT_EMA400 | -34.10 |
| SELL | 2025-08-26 12:15:00 | 568.65 | 2025-08-29 13:15:00 | 538.19 | TARGET | 30.46 |
| BUY | 2026-03-18 09:15:00 | 482.85 | 2026-03-24 09:15:00 | 452.00 | EXIT_EMA400 | -30.85 |
