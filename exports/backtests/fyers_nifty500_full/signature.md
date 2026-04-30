# Signatureglobal (India) Ltd. (SIGNATURE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 860.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 5 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / EMA400 exits:** 6 / 4
- **Total realized P&L (per unit):** 520.56
- **Avg P&L per closed trade:** 52.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1363.50 | 1476.80 | 1476.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1350.80 | 1456.99 | 1466.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1361.75 | 1354.91 | 1394.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 12:15:00 | 1351.10 | 1355.67 | 1393.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1368.05 | 1356.18 | 1392.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-05 10:15:00 | 1362.75 | 1356.24 | 1392.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1343.00 | 1304.17 | 1344.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-27 14:15:00 | 1351.00 | 1306.02 | 1344.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1232.20 | 1149.88 | 1149.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 1261.60 | 1157.27 | 1153.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1234.00 | 1240.71 | 1210.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 1249.10 | 1240.59 | 1212.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1230.20 | 1246.26 | 1229.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-22 10:15:00 | 1224.10 | 1246.04 | 1229.88 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1113.60 | 1217.56 | 1217.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1103.80 | 1190.84 | 1203.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 1124.30 | 1123.81 | 1151.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 11:15:00 | 1115.00 | 1124.33 | 1149.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 10:15:00 | 1153.10 | 1124.92 | 1148.35 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1126.70 | 1099.28 | 1099.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1134.00 | 1104.18 | 1101.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1104.30 | 1104.91 | 1102.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 1112.60 | 1104.98 | 1102.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1104.20 | 1105.96 | 1102.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-10 14:15:00 | 1111.40 | 1106.02 | 1102.96 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1117.40 | 1121.12 | 1113.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-31 12:15:00 | 1120.00 | 1120.96 | 1113.18 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1113.20 | 1120.88 | 1113.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-31 14:15:00 | 1127.80 | 1120.95 | 1113.25 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-07 09:15:00 | 1112.10 | 1122.45 | 1115.15 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 999.00 | 1108.17 | 1108.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 951.50 | 1104.65 | 1106.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 952.75 | 937.27 | 993.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-24 14:15:00 | 931.15 | 979.17 | 1001.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 995.45 | 973.05 | 995.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-02 09:15:00 | 954.40 | 972.86 | 995.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 864.60 | 818.14 | 866.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 11:15:00 | 868.80 | 821.79 | 866.69 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-05 10:15:00 | 1362.75 | 2024-12-09 09:15:00 | 1273.27 | TARGET | 89.48 |
| SELL | 2024-12-04 12:15:00 | 1351.10 | 2024-12-11 09:15:00 | 1224.77 | TARGET | 126.33 |
| BUY | 2025-06-24 11:15:00 | 1249.10 | 2025-07-22 10:15:00 | 1224.10 | EXIT_EMA400 | -25.00 |
| SELL | 2025-09-12 11:15:00 | 1115.00 | 2025-09-16 10:15:00 | 1153.10 | EXIT_EMA400 | -38.10 |
| BUY | 2025-12-10 14:15:00 | 1111.40 | 2025-12-18 10:15:00 | 1136.71 | TARGET | 25.31 |
| BUY | 2025-12-09 10:15:00 | 1112.60 | 2025-12-18 11:15:00 | 1143.54 | TARGET | 30.94 |
| BUY | 2025-12-31 12:15:00 | 1120.00 | 2026-01-07 09:15:00 | 1112.10 | EXIT_EMA400 | -7.90 |
| BUY | 2025-12-31 14:15:00 | 1127.80 | 2026-01-07 09:15:00 | 1112.10 | EXIT_EMA400 | -15.70 |
| SELL | 2026-03-02 09:15:00 | 954.40 | 2026-03-12 09:15:00 | 830.37 | TARGET | 124.03 |
| SELL | 2026-02-24 14:15:00 | 931.15 | 2026-03-30 10:15:00 | 719.98 | TARGET | 211.17 |
