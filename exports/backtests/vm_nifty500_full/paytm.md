# One 97 Communications Ltd. (PAYTM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1095.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 153.39
- **Avg P&L per closed trade:** 25.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 10:15:00 | 670.45 | 884.14 | 884.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 643.75 | 863.33 | 874.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 11:15:00 | 704.95 | 696.24 | 746.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-01 09:15:00 | 609.00 | 725.98 | 748.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 387.00 | 362.46 | 394.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-31 11:15:00 | 367.85 | 362.72 | 394.57 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-10 09:15:00 | 398.25 | 362.32 | 388.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 10:15:00 | 471.00 | 400.47 | 400.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 12:15:00 | 478.00 | 401.91 | 401.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 426.70 | 427.21 | 415.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-19 11:15:00 | 450.00 | 427.52 | 415.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 865.95 | 949.16 | 887.03 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 776.85 | 857.76 | 858.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 769.10 | 841.38 | 849.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 743.75 | 736.65 | 773.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 721.15 | 738.26 | 772.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 12:15:00 | 777.95 | 739.68 | 770.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 864.65 | 787.04 | 786.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 878.30 | 793.78 | 790.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 830.10 | 830.99 | 812.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 09:15:00 | 861.65 | 831.39 | 813.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 817.95 | 832.76 | 815.08 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-07 09:15:00 | 861.65 | 833.05 | 815.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 825.55 | 843.42 | 827.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 1171.80 | 1279.96 | 1280.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1160.00 | 1272.59 | 1276.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1074.35 | 1051.37 | 1106.58 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-01 09:15:00 | 609.00 | 2024-06-10 09:15:00 | 398.25 | EXIT_EMA400 | 210.75 |
| SELL | 2024-05-31 11:15:00 | 367.85 | 2024-06-10 09:15:00 | 398.25 | EXIT_EMA400 | -30.40 |
| BUY | 2024-07-19 11:15:00 | 450.00 | 2024-08-16 09:15:00 | 552.04 | TARGET | 102.04 |
| SELL | 2025-03-20 09:15:00 | 721.15 | 2025-03-24 12:15:00 | 777.95 | EXIT_EMA400 | -56.80 |
| BUY | 2025-05-05 09:15:00 | 861.65 | 2025-05-22 12:15:00 | 825.55 | EXIT_EMA400 | -36.10 |
| BUY | 2025-05-07 09:15:00 | 861.65 | 2025-05-22 12:15:00 | 825.55 | EXIT_EMA400 | -36.10 |
