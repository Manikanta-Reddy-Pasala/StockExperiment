# Tejas Networks Ltd. (TEJASNET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4998 bars)
- **Last close:** 415.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 8 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / EMA400 exits:** 3 / 10
- **Total realized P&L (per unit):** 378.53
- **Avg P&L per closed trade:** 29.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 13:15:00 | 817.85 | 845.09 | 845.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 15:15:00 | 813.10 | 844.49 | 844.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 11:15:00 | 841.90 | 831.84 | 837.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-11 09:15:00 | 829.00 | 831.89 | 837.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-18 09:15:00 | 835.40 | 828.58 | 834.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 863.40 | 838.82 | 838.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 14:15:00 | 866.80 | 840.50 | 839.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 849.10 | 852.45 | 847.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-17 10:15:00 | 857.95 | 852.33 | 847.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 857.95 | 852.33 | 847.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-19 10:15:00 | 865.00 | 852.25 | 847.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-23 09:15:00 | 782.95 | 852.11 | 847.63 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 13:15:00 | 784.05 | 843.26 | 843.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 776.50 | 839.74 | 841.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 762.35 | 759.13 | 783.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-05 10:15:00 | 749.55 | 758.95 | 782.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-02 14:15:00 | 756.65 | 717.94 | 745.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 1088.25 | 764.30 | 763.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 13:15:00 | 1090.65 | 797.37 | 780.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1052.35 | 1096.17 | 1001.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 1171.60 | 1096.03 | 1007.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1295.45 | 1366.46 | 1266.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-23 12:15:00 | 1265.00 | 1364.75 | 1266.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 1216.00 | 1263.96 | 1264.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 1213.15 | 1263.06 | 1263.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1348.10 | 1210.50 | 1230.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 11:15:00 | 1276.05 | 1212.46 | 1231.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1276.05 | 1212.46 | 1231.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-21 12:15:00 | 1299.05 | 1213.32 | 1231.76 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 14:15:00 | 1317.45 | 1246.04 | 1245.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1361.50 | 1251.83 | 1248.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1261.95 | 1283.87 | 1266.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-19 09:15:00 | 1308.95 | 1280.96 | 1266.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1268.70 | 1282.45 | 1268.17 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-21 14:15:00 | 1267.20 | 1282.09 | 1268.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 1193.00 | 1281.89 | 1281.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 1190.85 | 1277.12 | 1279.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 809.85 | 767.55 | 876.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-30 10:15:00 | 705.50 | 814.43 | 846.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 623.45 | 602.02 | 644.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 12:15:00 | 592.40 | 604.14 | 639.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 620.10 | 598.59 | 623.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 14:15:00 | 611.65 | 602.26 | 622.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 599.30 | 599.41 | 617.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-29 10:15:00 | 594.20 | 599.36 | 617.43 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-27 09:15:00 | 407.90 | 350.16 | 389.14 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 432.75 | 414.82 | 414.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 434.95 | 415.19 | 414.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 411.40 | 415.36 | 415.01 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 400.15 | 414.64 | 414.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 426.00 | 414.75 | 414.72 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 395.75 | 414.61 | 414.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 392.30 | 414.38 | 414.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 417.50 | 413.75 | 414.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-02 09:15:00 | 406.20 | 413.91 | 414.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-02 13:15:00 | 417.60 | 413.78 | 414.24 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 421.75 | 414.73 | 414.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 436.20 | 415.08 | 414.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 423.70 | 424.07 | 419.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-29 10:15:00 | 427.00 | 419.05 | 418.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 427.00 | 419.05 | 418.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-29 13:15:00 | 417.45 | 419.05 | 418.06 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-11 09:15:00 | 829.00 | 2023-12-18 09:15:00 | 835.40 | EXIT_EMA400 | -6.40 |
| BUY | 2024-01-17 10:15:00 | 857.95 | 2024-01-23 09:15:00 | 782.95 | EXIT_EMA400 | -75.00 |
| BUY | 2024-01-19 10:15:00 | 865.00 | 2024-01-23 09:15:00 | 782.95 | EXIT_EMA400 | -82.05 |
| SELL | 2024-03-05 10:15:00 | 749.55 | 2024-04-02 14:15:00 | 756.65 | EXIT_EMA400 | -7.10 |
| BUY | 2024-06-06 09:15:00 | 1171.60 | 2024-07-23 12:15:00 | 1265.00 | EXIT_EMA400 | 93.40 |
| SELL | 2024-10-21 11:15:00 | 1276.05 | 2024-10-21 12:15:00 | 1299.05 | EXIT_EMA400 | -23.00 |
| BUY | 2024-11-19 09:15:00 | 1308.95 | 2024-11-21 14:15:00 | 1267.20 | EXIT_EMA400 | -41.75 |
| SELL | 2025-09-19 14:15:00 | 611.65 | 2025-09-26 09:15:00 | 579.24 | TARGET | 32.41 |
| SELL | 2025-09-29 10:15:00 | 594.20 | 2025-11-07 09:15:00 | 524.52 | TARGET | 69.68 |
| SELL | 2025-08-28 12:15:00 | 592.40 | 2025-12-18 09:15:00 | 450.72 | TARGET | 141.69 |
| SELL | 2025-04-30 10:15:00 | 705.50 | 2026-02-27 09:15:00 | 407.90 | EXIT_EMA400 | 297.60 |
| SELL | 2026-04-02 09:15:00 | 406.20 | 2026-04-02 13:15:00 | 417.60 | EXIT_EMA400 | -11.40 |
| BUY | 2026-04-29 10:15:00 | 427.00 | 2026-04-29 13:15:00 | 417.45 | EXIT_EMA400 | -9.55 |
