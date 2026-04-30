# Fortis Healthcare Ltd. (FORTIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 922.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 19.07
- **Avg P&L per closed trade:** 2.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 10:15:00 | 401.00 | 408.96 | 408.99 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 10:15:00 | 416.25 | 409.01 | 409.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 420.15 | 409.20 | 409.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 437.90 | 439.33 | 429.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-13 12:15:00 | 445.45 | 439.37 | 429.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 434.45 | 451.75 | 441.22 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 10:15:00 | 636.70 | 655.67 | 655.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 632.55 | 655.44 | 655.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 661.00 | 651.11 | 653.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 13:15:00 | 646.85 | 651.86 | 653.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 649.45 | 651.81 | 653.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-07 13:15:00 | 647.70 | 651.70 | 653.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 641.40 | 651.53 | 653.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 10:15:00 | 629.90 | 651.31 | 653.22 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-04 09:15:00 | 638.45 | 624.96 | 635.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 687.75 | 638.23 | 638.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 691.85 | 654.41 | 648.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 662.65 | 662.95 | 653.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 674.00 | 662.91 | 654.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 991.20 | 1026.17 | 987.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-10 14:15:00 | 994.50 | 1025.86 | 987.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 993.00 | 1025.20 | 987.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-11 10:15:00 | 982.50 | 1024.78 | 987.10 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 908.10 | 965.76 | 965.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 905.05 | 965.16 | 965.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 13:15:00 | 912.50 | 907.45 | 929.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 10:15:00 | 901.50 | 907.51 | 928.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 12:15:00 | 922.50 | 902.19 | 921.11 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 11:15:00 | 949.50 | 903.90 | 903.75 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 883.55 | 904.16 | 904.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 14:15:00 | 879.45 | 903.92 | 904.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 845.30 | 842.32 | 865.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 10:15:00 | 842.55 | 842.32 | 865.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 867.30 | 844.33 | 863.34 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 952.65 | 876.66 | 876.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 958.00 | 880.29 | 878.27 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-13 12:15:00 | 445.45 | 2024-06-04 10:15:00 | 434.45 | EXIT_EMA400 | -11.00 |
| SELL | 2025-02-07 13:15:00 | 647.70 | 2025-02-10 10:15:00 | 630.45 | TARGET | 17.25 |
| SELL | 2025-02-06 13:15:00 | 646.85 | 2025-02-10 12:15:00 | 626.64 | TARGET | 20.21 |
| SELL | 2025-02-10 10:15:00 | 629.90 | 2025-03-04 09:15:00 | 638.45 | EXIT_EMA400 | -8.55 |
| BUY | 2025-05-12 09:15:00 | 674.00 | 2025-05-22 11:15:00 | 732.92 | TARGET | 58.92 |
| BUY | 2025-11-10 14:15:00 | 994.50 | 2025-11-11 10:15:00 | 982.50 | EXIT_EMA400 | -12.00 |
| SELL | 2025-12-23 10:15:00 | 901.50 | 2026-01-02 12:15:00 | 922.50 | EXIT_EMA400 | -21.00 |
| SELL | 2026-04-08 10:15:00 | 842.55 | 2026-04-15 09:15:00 | 867.30 | EXIT_EMA400 | -24.75 |
