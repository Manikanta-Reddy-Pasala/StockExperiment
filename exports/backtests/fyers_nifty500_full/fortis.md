# Fortis Healthcare Ltd. (FORTIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 922.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 53.20
- **Avg P&L per closed trade:** 8.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 09:15:00 | 641.00 | 655.87 | 655.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 632.55 | 655.44 | 655.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 661.00 | 649.71 | 652.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 13:15:00 | 646.85 | 650.57 | 652.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 646.85 | 650.57 | 652.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 09:15:00 | 641.70 | 650.39 | 652.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-04 09:15:00 | 638.45 | 624.53 | 635.43 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 686.25 | 637.60 | 637.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 691.85 | 654.36 | 648.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 662.65 | 662.88 | 653.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 674.10 | 662.85 | 654.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 991.20 | 1026.17 | 987.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-10 14:15:00 | 994.30 | 1025.85 | 987.06 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 993.00 | 1025.20 | 987.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-11 10:15:00 | 982.50 | 1024.77 | 987.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 908.15 | 965.74 | 965.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 905.20 | 965.14 | 965.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 907.45 | 907.37 | 929.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 10:15:00 | 901.50 | 907.45 | 928.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 12:15:00 | 922.85 | 902.18 | 921.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 956.15 | 902.29 | 902.29 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 876.85 | 902.88 | 902.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 876.20 | 902.62 | 902.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 845.30 | 842.24 | 864.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 10:15:00 | 842.55 | 842.24 | 864.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 866.85 | 844.29 | 863.01 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 957.00 | 876.17 | 875.83 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 13:15:00 | 646.85 | 2025-02-10 10:15:00 | 628.74 | TARGET | 18.11 |
| SELL | 2025-02-10 09:15:00 | 641.70 | 2025-02-12 09:15:00 | 608.74 | TARGET | 32.96 |
| BUY | 2025-05-12 09:15:00 | 674.10 | 2025-05-22 11:15:00 | 733.68 | TARGET | 59.58 |
| BUY | 2025-11-10 14:15:00 | 994.30 | 2025-11-11 10:15:00 | 982.50 | EXIT_EMA400 | -11.80 |
| SELL | 2025-12-23 10:15:00 | 901.50 | 2026-01-02 12:15:00 | 922.85 | EXIT_EMA400 | -21.35 |
| SELL | 2026-04-08 10:15:00 | 842.55 | 2026-04-15 09:15:00 | 866.85 | EXIT_EMA400 | -24.30 |
