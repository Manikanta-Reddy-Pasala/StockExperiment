# HDFC Bank Ltd. (HDFCBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 774.75
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
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 79.20
- **Avg P&L per closed trade:** 11.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 10:15:00 | 826.45 | 876.41 | 876.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 11:15:00 | 821.00 | 875.86 | 876.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 851.55 | 848.53 | 859.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 11:15:00 | 843.00 | 848.53 | 858.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-04 12:15:00 | 858.30 | 847.94 | 857.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 886.00 | 856.78 | 856.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 897.25 | 857.18 | 856.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.76 | 869.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 12:15:00 | 890.93 | 878.84 | 870.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 990.40 | 997.71 | 983.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-07 14:15:00 | 997.70 | 996.64 | 983.92 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 985.00 | 995.54 | 984.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-13 13:15:00 | 991.10 | 995.03 | 984.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-22 09:15:00 | 982.65 | 995.34 | 986.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 958.70 | 980.47 | 980.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 947.40 | 971.43 | 974.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 968.95 | 965.15 | 970.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-03 09:15:00 | 957.75 | 965.04 | 970.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1000.55 | 974.00 | 973.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.35 | 974.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.98 | 981.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 10:15:00 | 990.00 | 986.47 | 981.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 981.05 | 986.41 | 982.05 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 944.15 | 947.54 | 960.84 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-01 11:15:00 | 843.00 | 2025-02-04 12:15:00 | 858.30 | EXIT_EMA400 | -15.30 |
| BUY | 2025-04-08 12:15:00 | 890.93 | 2025-04-17 12:15:00 | 953.63 | TARGET | 62.70 |
| BUY | 2025-08-13 13:15:00 | 991.10 | 2025-08-18 09:15:00 | 1010.38 | TARGET | 19.28 |
| BUY | 2025-08-07 14:15:00 | 997.70 | 2025-08-22 09:15:00 | 982.65 | EXIT_EMA400 | -15.05 |
| SELL | 2025-10-03 09:15:00 | 957.75 | 2025-10-06 09:15:00 | 971.30 | EXIT_EMA400 | -13.55 |
| BUY | 2025-11-10 10:15:00 | 990.00 | 2025-11-11 09:15:00 | 981.05 | EXIT_EMA400 | -8.95 |
| SELL | 2026-02-06 09:15:00 | 944.15 | 2026-02-27 09:15:00 | 894.08 | TARGET | 50.07 |
