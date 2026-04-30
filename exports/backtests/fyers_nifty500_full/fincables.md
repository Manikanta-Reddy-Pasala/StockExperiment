# Finolex Cables Ltd. (FINCABLES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 990.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 87.43
- **Avg P&L per closed trade:** 17.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 1414.00 | 1451.51 | 1451.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1407.00 | 1449.80 | 1450.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 1443.00 | 1435.82 | 1443.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 11:15:00 | 1427.80 | 1444.93 | 1447.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 11:15:00 | 1266.00 | 1192.98 | 1254.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 983.55 | 922.63 | 922.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 995.00 | 925.17 | 923.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 09:15:00 | 973.75 | 949.99 | 943.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 954.00 | 958.85 | 950.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 950.05 | 958.62 | 950.80 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 903.10 | 945.77 | 945.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 893.45 | 943.46 | 944.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 863.80 | 862.32 | 889.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-02 13:15:00 | 855.00 | 862.25 | 889.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 778.40 | 757.67 | 780.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-17 10:15:00 | 771.80 | 758.40 | 780.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 814.50 | 767.78 | 767.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 816.85 | 769.11 | 768.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 852.30 | 853.01 | 820.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 15:15:00 | 861.40 | 852.41 | 822.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 11:15:00 | 824.80 | 854.77 | 826.93 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 11:15:00 | 1427.80 | 2024-10-03 13:15:00 | 1370.05 | TARGET | 57.75 |
| BUY | 2025-06-30 09:15:00 | 973.75 | 2025-07-14 12:15:00 | 950.05 | EXIT_EMA400 | -23.70 |
| SELL | 2025-09-02 13:15:00 | 855.00 | 2025-11-24 09:15:00 | 752.66 | TARGET | 102.34 |
| SELL | 2025-12-17 10:15:00 | 771.80 | 2025-12-19 09:15:00 | 784.15 | EXIT_EMA400 | -12.35 |
| BUY | 2026-03-17 15:15:00 | 861.40 | 2026-03-23 11:15:00 | 824.80 | EXIT_EMA400 | -36.60 |
