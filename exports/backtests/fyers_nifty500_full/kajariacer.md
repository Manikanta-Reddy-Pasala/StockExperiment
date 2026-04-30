# Kajaria Ceramics Ltd. (KAJARIACER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1180.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 271.59
- **Avg P&L per closed trade:** 67.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 1274.05 | 1408.26 | 1408.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 1264.65 | 1406.83 | 1408.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 14:15:00 | 1172.80 | 1171.58 | 1218.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-02 11:15:00 | 1150.00 | 1170.98 | 1217.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 855.95 | 829.01 | 870.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-12 10:15:00 | 847.60 | 829.20 | 869.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-14 10:15:00 | 883.85 | 832.22 | 868.73 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1047.65 | 894.90 | 894.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1050.95 | 916.59 | 905.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 11:15:00 | 1223.90 | 1228.54 | 1168.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-28 13:15:00 | 1233.40 | 1228.61 | 1169.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1204.00 | 1229.13 | 1195.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-23 12:15:00 | 1188.00 | 1226.35 | 1195.86 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1115.00 | 1198.35 | 1198.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1106.60 | 1188.43 | 1193.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1092.50 | 1091.89 | 1127.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 10:15:00 | 1073.10 | 1091.27 | 1123.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-18 09:15:00 | 1007.60 | 952.62 | 991.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1115.05 | 974.83 | 974.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 1127.40 | 981.76 | 978.19 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-02 11:15:00 | 1150.00 | 2025-02-11 09:15:00 | 947.76 | TARGET | 202.24 |
| SELL | 2025-05-12 10:15:00 | 847.60 | 2025-05-14 10:15:00 | 883.85 | EXIT_EMA400 | -36.25 |
| BUY | 2025-08-28 13:15:00 | 1233.40 | 2025-09-23 12:15:00 | 1188.00 | EXIT_EMA400 | -45.40 |
| SELL | 2025-12-18 10:15:00 | 1073.10 | 2026-01-27 09:15:00 | 922.09 | TARGET | 151.01 |
