# UPL Ltd. (UPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 642.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 35.05
- **Avg P&L per closed trade:** 5.01

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 513.00 | 554.04 | 554.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 509.74 | 553.60 | 553.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 539.81 | 538.66 | 544.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 532.48 | 539.12 | 544.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-21 12:15:00 | 538.09 | 529.03 | 537.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 555.45 | 543.03 | 542.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 563.90 | 544.33 | 543.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 543.85 | 545.07 | 544.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-12 14:15:00 | 548.65 | 545.15 | 544.12 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-13 09:15:00 | 538.70 | 545.11 | 544.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 13:15:00 | 519.15 | 543.24 | 543.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 512.35 | 542.46 | 542.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 12:15:00 | 529.00 | 524.31 | 532.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 13:15:00 | 519.25 | 524.54 | 532.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-07 09:15:00 | 539.65 | 524.61 | 531.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 554.50 | 536.56 | 536.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 556.75 | 536.94 | 536.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-27 13:15:00 | 541.70 | 538.68 | 537.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 539.00 | 538.74 | 537.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-29 09:15:00 | 547.90 | 539.03 | 537.90 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 606.15 | 634.45 | 615.50 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.12 | 688.12 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 741.00 | 724.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 748.00 | 740.93 | 725.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.21 | 757.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 680.75 | 744.85 | 745.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 663.85 | 744.04 | 744.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 12:15:00 | 736.80 | 739.27 | 742.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 736.80 | 739.27 | 742.08 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 757.70 | 739.45 | 742.11 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 15:15:00 | 532.48 | 2024-11-11 14:15:00 | 496.01 | TARGET | 36.47 |
| BUY | 2024-12-12 14:15:00 | 548.65 | 2024-12-13 09:15:00 | 538.70 | EXIT_EMA400 | -9.95 |
| SELL | 2025-01-06 13:15:00 | 519.25 | 2025-01-07 09:15:00 | 539.65 | EXIT_EMA400 | -20.40 |
| BUY | 2025-01-27 13:15:00 | 541.70 | 2025-01-29 13:15:00 | 553.78 | TARGET | 12.08 |
| BUY | 2025-01-29 09:15:00 | 547.90 | 2025-01-30 12:15:00 | 577.90 | TARGET | 30.00 |
| BUY | 2025-12-10 09:15:00 | 748.00 | 2026-01-20 10:15:00 | 755.75 | EXIT_EMA400 | 7.75 |
| SELL | 2026-02-03 12:15:00 | 736.80 | 2026-02-04 09:15:00 | 757.70 | EXIT_EMA400 | -20.90 |
