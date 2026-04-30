# Sun TV Network Ltd. (SUNTV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 609.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -83.45
- **Avg P&L per closed trade:** -10.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 759.70 | 802.77 | 802.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 751.70 | 801.83 | 802.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 762.05 | 752.92 | 768.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 09:15:00 | 747.30 | 753.52 | 768.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 765.55 | 753.56 | 766.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 773.30 | 753.76 | 766.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 671.65 | 633.51 | 633.38 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 597.70 | 634.91 | 634.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 586.10 | 613.57 | 620.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 591.20 | 590.04 | 603.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 12:15:00 | 587.70 | 590.24 | 603.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 567.55 | 574.70 | 589.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 14:15:00 | 554.60 | 575.43 | 585.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 571.70 | 569.85 | 581.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 11:15:00 | 547.65 | 562.06 | 571.81 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 558.75 | 552.52 | 564.76 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-01 12:15:00 | 567.15 | 552.67 | 564.77 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 574.20 | 560.34 | 560.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 584.35 | 560.58 | 560.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 540.35 | 561.90 | 561.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 539.25 | 561.06 | 561.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-27 09:15:00 | 529.90 | 558.68 | 560.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 556.00 | 553.94 | 557.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-01 15:15:00 | 545.85 | 553.77 | 557.21 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 549.10 | 552.98 | 556.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 14:15:00 | 544.70 | 552.80 | 556.49 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 554.70 | 550.07 | 554.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 610.55 | 558.89 | 558.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 635.55 | 575.72 | 568.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 15:15:00 | 604.00 | 581.00 | 575.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 582.15 | 587.29 | 579.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-30 10:15:00 | 574.75 | 587.18 | 579.76 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 09:15:00 | 747.30 | 2024-12-03 11:15:00 | 773.30 | EXIT_EMA400 | -26.00 |
| SELL | 2025-07-22 12:15:00 | 587.70 | 2025-08-29 09:15:00 | 540.20 | TARGET | 47.50 |
| SELL | 2025-08-26 14:15:00 | 554.60 | 2025-10-01 12:15:00 | 567.15 | EXIT_EMA400 | -12.55 |
| SELL | 2025-09-23 11:15:00 | 547.65 | 2025-10-01 12:15:00 | 567.15 | EXIT_EMA400 | -19.50 |
| SELL | 2026-01-27 09:15:00 | 529.90 | 2026-02-09 09:15:00 | 554.70 | EXIT_EMA400 | -24.80 |
| SELL | 2026-02-01 15:15:00 | 545.85 | 2026-02-09 09:15:00 | 554.70 | EXIT_EMA400 | -8.85 |
| SELL | 2026-02-03 14:15:00 | 544.70 | 2026-02-09 09:15:00 | 554.70 | EXIT_EMA400 | -10.00 |
| BUY | 2026-03-18 15:15:00 | 604.00 | 2026-03-30 10:15:00 | 574.75 | EXIT_EMA400 | -29.25 |
