# Bikaji Foods International Ltd. (BIKAJI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 676.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -86.53
- **Avg P&L per closed trade:** -12.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 522.50 | 545.88 | 545.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 514.75 | 545.57 | 545.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 10:15:00 | 536.05 | 535.53 | 540.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-11 11:15:00 | 527.25 | 535.45 | 540.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 522.00 | 509.43 | 522.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-02 15:15:00 | 526.00 | 509.59 | 522.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 539.40 | 527.07 | 527.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 566.40 | 529.10 | 528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 882.95 | 896.87 | 841.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-16 14:15:00 | 910.00 | 883.63 | 846.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 11:15:00 | 847.35 | 886.76 | 852.93 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 729.45 | 843.61 | 843.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 725.10 | 787.91 | 805.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 737.00 | 712.94 | 748.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 09:15:00 | 657.05 | 715.97 | 745.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-17 11:15:00 | 685.00 | 657.56 | 685.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 730.55 | 690.22 | 690.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 740.60 | 690.72 | 690.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 699.30 | 699.95 | 695.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 706.10 | 697.99 | 695.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 706.10 | 697.99 | 695.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 12:15:00 | 693.90 | 697.98 | 695.15 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 730.00 | 756.68 | 756.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 724.50 | 755.16 | 755.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 763.00 | 748.98 | 752.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-15 09:15:00 | 736.80 | 748.86 | 752.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 736.80 | 748.86 | 752.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-15 10:15:00 | 734.85 | 748.72 | 752.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 724.80 | 716.04 | 725.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-10 14:15:00 | 730.30 | 716.44 | 725.86 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 756.55 | 731.57 | 731.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 13:15:00 | 761.10 | 731.86 | 731.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 734.55 | 736.07 | 733.99 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 712.70 | 732.05 | 732.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 710.65 | 731.46 | 731.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 632.00 | 630.69 | 653.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-23 09:15:00 | 612.30 | 630.50 | 653.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 643.20 | 626.12 | 644.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 11:15:00 | 645.50 | 627.38 | 644.34 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 675.00 | 652.84 | 652.75 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-11 11:15:00 | 527.25 | 2024-03-14 14:15:00 | 488.83 | TARGET | 38.42 |
| BUY | 2024-10-16 14:15:00 | 910.00 | 2024-10-22 11:15:00 | 847.35 | EXIT_EMA400 | -62.65 |
| SELL | 2025-02-07 09:15:00 | 657.05 | 2025-03-17 11:15:00 | 685.00 | EXIT_EMA400 | -27.95 |
| BUY | 2025-05-08 09:15:00 | 706.10 | 2025-05-08 12:15:00 | 693.90 | EXIT_EMA400 | -12.20 |
| SELL | 2025-10-15 09:15:00 | 736.80 | 2025-12-10 14:15:00 | 730.30 | EXIT_EMA400 | 6.50 |
| SELL | 2025-10-15 10:15:00 | 734.85 | 2025-12-10 14:15:00 | 730.30 | EXIT_EMA400 | 4.55 |
| SELL | 2026-03-23 09:15:00 | 612.30 | 2026-04-09 11:15:00 | 645.50 | EXIT_EMA400 | -33.20 |
