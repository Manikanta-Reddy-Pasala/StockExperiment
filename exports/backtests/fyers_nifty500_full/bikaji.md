# Bikaji Foods International Ltd. (BIKAJI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 675.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -62.40
- **Avg P&L per closed trade:** -12.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 729.45 | 843.48 | 843.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 724.15 | 786.07 | 804.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 732.20 | 712.73 | 747.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 09:15:00 | 657.00 | 716.36 | 744.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-17 11:15:00 | 685.00 | 657.51 | 684.54 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 14:15:00 | 729.85 | 689.79 | 689.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 15:15:00 | 731.90 | 690.21 | 689.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 699.30 | 699.96 | 695.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 706.10 | 698.02 | 694.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 706.10 | 698.02 | 694.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 12:15:00 | 693.90 | 698.01 | 695.03 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 733.20 | 756.52 | 756.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 724.50 | 755.22 | 755.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-15 09:15:00 | 736.80 | 748.88 | 752.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 736.80 | 748.88 | 752.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-15 10:15:00 | 734.85 | 748.74 | 752.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 724.80 | 715.99 | 725.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-10 14:15:00 | 730.30 | 716.39 | 725.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 13:15:00 | 761.30 | 731.82 | 731.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 712.70 | 732.07 | 732.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 707.40 | 731.44 | 731.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 632.00 | 630.53 | 653.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-23 09:15:00 | 612.30 | 630.37 | 652.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 643.20 | 626.01 | 644.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 11:15:00 | 645.55 | 627.28 | 644.02 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 675.65 | 652.29 | 652.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 677.25 | 652.98 | 652.58 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-07 09:15:00 | 657.00 | 2025-03-17 11:15:00 | 685.00 | EXIT_EMA400 | -28.00 |
| BUY | 2025-05-08 09:15:00 | 706.10 | 2025-05-08 12:15:00 | 693.90 | EXIT_EMA400 | -12.20 |
| SELL | 2025-10-15 09:15:00 | 736.80 | 2025-12-10 14:15:00 | 730.30 | EXIT_EMA400 | 6.50 |
| SELL | 2025-10-15 10:15:00 | 734.85 | 2025-12-10 14:15:00 | 730.30 | EXIT_EMA400 | 4.55 |
| SELL | 2026-03-23 09:15:00 | 612.30 | 2026-04-09 11:15:00 | 645.55 | EXIT_EMA400 | -33.25 |
