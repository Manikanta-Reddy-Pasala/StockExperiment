# Aditya Birla Sun Life AMC Ltd. (ABSLAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1015.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 32.16
- **Avg P&L per closed trade:** 8.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 13:15:00 | 461.00 | 479.18 | 479.25 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 495.55 | 479.18 | 479.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 11:15:00 | 499.25 | 479.38 | 479.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 15:15:00 | 524.00 | 524.44 | 509.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-22 11:15:00 | 528.65 | 523.87 | 510.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-31 15:15:00 | 515.10 | 527.04 | 515.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 755.65 | 795.11 | 795.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 739.20 | 793.71 | 794.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 636.75 | 635.81 | 674.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 622.25 | 640.89 | 665.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 656.75 | 636.15 | 657.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 659.40 | 636.97 | 657.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 716.80 | 662.73 | 662.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 722.00 | 672.79 | 668.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 736.25 | 741.49 | 714.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 748.65 | 741.51 | 715.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 833.75 | 855.02 | 828.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-02 14:15:00 | 827.25 | 853.26 | 829.16 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 799.70 | 824.75 | 824.76 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 852.30 | 824.58 | 824.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 11:15:00 | 860.90 | 824.94 | 824.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 830.45 | 837.52 | 831.79 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 773.15 | 826.59 | 826.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 771.10 | 826.04 | 826.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 753.70 | 745.23 | 768.53 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 853.65 | 777.43 | 777.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 856.30 | 778.22 | 777.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 790.20 | 791.52 | 784.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 15:15:00 | 805.00 | 791.26 | 784.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 13:15:00 | 787.30 | 798.43 | 789.81 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-22 11:15:00 | 528.65 | 2024-05-31 15:15:00 | 515.10 | EXIT_EMA400 | -13.55 |
| SELL | 2025-04-07 09:15:00 | 622.25 | 2025-04-22 09:15:00 | 659.40 | EXIT_EMA400 | -37.15 |
| BUY | 2025-06-20 09:15:00 | 748.65 | 2025-07-10 12:15:00 | 849.21 | TARGET | 100.56 |
| BUY | 2026-01-12 15:15:00 | 805.00 | 2026-01-20 13:15:00 | 787.30 | EXIT_EMA400 | -17.70 |
