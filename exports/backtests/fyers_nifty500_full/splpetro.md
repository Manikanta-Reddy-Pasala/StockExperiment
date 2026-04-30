# Supreme Petrochem Ltd. (SPLPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 755.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 76.46
- **Avg P&L per closed trade:** 19.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 781.85 | 838.48 | 838.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 779.70 | 837.90 | 838.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 743.00 | 735.28 | 765.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 09:15:00 | 718.00 | 748.86 | 764.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 667.60 | 637.50 | 673.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-05 15:15:00 | 677.00 | 639.19 | 672.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 645.25 | 624.24 | 624.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 648.35 | 624.48 | 624.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 804.45 | 804.92 | 767.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-23 09:15:00 | 817.75 | 805.22 | 769.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 772.80 | 806.23 | 781.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 754.85 | 796.30 | 796.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 749.10 | 795.01 | 795.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 679.45 | 669.49 | 707.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 09:15:00 | 635.85 | 666.58 | 700.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 630.80 | 576.61 | 615.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 682.00 | 630.51 | 630.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 692.25 | 637.16 | 633.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 652.65 | 653.00 | 643.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 11:15:00 | 666.90 | 653.61 | 643.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 668.45 | 655.26 | 645.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 643.25 | 655.51 | 646.18 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 09:15:00 | 718.00 | 2025-01-27 09:15:00 | 577.99 | TARGET | 140.01 |
| BUY | 2025-07-23 09:15:00 | 817.75 | 2025-08-07 10:15:00 | 772.80 | EXIT_EMA400 | -44.95 |
| SELL | 2025-12-16 09:15:00 | 635.85 | 2026-02-03 11:15:00 | 630.80 | EXIT_EMA400 | 5.05 |
| BUY | 2026-03-17 11:15:00 | 666.90 | 2026-03-23 09:15:00 | 643.25 | EXIT_EMA400 | -23.65 |
