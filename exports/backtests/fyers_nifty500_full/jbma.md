# JBM Auto Ltd. (JBMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 628.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -203.47
- **Avg P&L per closed trade:** -29.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 929.25 | 1004.35 | 1004.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 927.90 | 1003.59 | 1004.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 14:15:00 | 976.83 | 976.38 | 987.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 09:15:00 | 970.53 | 976.32 | 987.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 974.85 | 970.96 | 982.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-06 09:15:00 | 956.05 | 971.04 | 982.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 964.45 | 967.54 | 978.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-11 15:15:00 | 958.50 | 967.45 | 978.73 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-12 09:15:00 | 1002.55 | 967.80 | 978.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 686.00 | 647.60 | 647.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 707.35 | 655.63 | 651.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 704.60 | 707.32 | 688.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 14:15:00 | 722.45 | 706.48 | 689.51 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 695.60 | 705.89 | 690.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 690.10 | 705.73 | 690.52 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 643.20 | 679.68 | 679.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 640.00 | 673.70 | 676.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 658.30 | 654.93 | 663.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 12:15:00 | 640.85 | 655.34 | 663.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 647.50 | 653.75 | 662.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-31 14:15:00 | 636.75 | 652.96 | 661.38 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-19 11:15:00 | 646.80 | 631.23 | 645.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 754.35 | 644.98 | 644.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 765.85 | 666.12 | 656.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 14:15:00 | 678.10 | 678.33 | 664.48 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 632.00 | 660.78 | 660.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 630.60 | 659.97 | 660.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 576.50 | 612.19 | 618.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 567.35 | 583.51 | 599.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 609.05 | 583.13 | 598.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 12:15:00 | 619.60 | 570.37 | 570.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 628.25 | 572.52 | 571.39 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-29 09:15:00 | 970.53 | 2024-09-12 09:15:00 | 1002.55 | EXIT_EMA400 | -32.02 |
| SELL | 2024-09-06 09:15:00 | 956.05 | 2024-09-12 09:15:00 | 1002.55 | EXIT_EMA400 | -46.50 |
| SELL | 2024-09-11 15:15:00 | 958.50 | 2024-09-12 09:15:00 | 1002.55 | EXIT_EMA400 | -44.05 |
| BUY | 2025-06-16 14:15:00 | 722.45 | 2025-06-19 10:15:00 | 690.10 | EXIT_EMA400 | -32.35 |
| SELL | 2025-07-28 12:15:00 | 640.85 | 2025-08-19 11:15:00 | 646.80 | EXIT_EMA400 | -5.95 |
| SELL | 2025-07-31 14:15:00 | 636.75 | 2025-08-19 11:15:00 | 646.80 | EXIT_EMA400 | -10.05 |
| SELL | 2026-01-20 09:15:00 | 576.50 | 2026-02-04 09:15:00 | 609.05 | EXIT_EMA400 | -32.55 |
