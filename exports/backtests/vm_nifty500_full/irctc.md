# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 539.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 64.89
- **Avg P&L per closed trade:** 21.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 991.45 | 1006.40 | 1006.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 13:15:00 | 988.40 | 1006.22 | 1006.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 948.80 | 945.95 | 964.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 09:15:00 | 933.30 | 945.77 | 963.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 853.10 | 831.95 | 853.81 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-11 14:15:00 | 855.95 | 832.58 | 853.80 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 765.35 | 744.57 | 744.53 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 738.45 | 744.47 | 744.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 732.50 | 744.35 | 744.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 749.90 | 742.40 | 743.43 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 13:15:00 | 764.25 | 744.46 | 744.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 773.05 | 745.12 | 744.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 764.70 | 769.22 | 759.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 10:15:00 | 781.95 | 769.10 | 760.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 770.80 | 774.02 | 765.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 762.00 | 773.73 | 765.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 729.15 | 766.99 | 767.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 724.75 | 763.31 | 765.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 727.95 | 722.98 | 734.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 721.75 | 725.67 | 733.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-20 09:15:00 | 727.10 | 715.42 | 723.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-06 09:15:00 | 933.30 | 2024-10-22 09:15:00 | 843.11 | TARGET | 90.19 |
| BUY | 2025-06-05 10:15:00 | 781.95 | 2025-06-16 09:15:00 | 762.00 | EXIT_EMA400 | -19.95 |
| SELL | 2025-09-23 10:15:00 | 721.75 | 2025-10-20 09:15:00 | 727.10 | EXIT_EMA400 | -5.35 |
