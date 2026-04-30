# Adani Green Energy Ltd. (ADANIGREEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1227.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 7 |
| ENTRY1 | 9 |
| ENTRY2 | 5 |
| EXIT | 9 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 2 / 12
- **Target hits / EMA400 exits:** 2 / 12
- **Total realized P&L (per unit):** -561.45
- **Avg P&L per closed trade:** -40.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 14:15:00 | 959.00 | 982.76 | 982.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 09:15:00 | 956.15 | 982.25 | 982.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 13:15:00 | 990.50 | 980.50 | 981.67 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 1004.50 | 982.96 | 982.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 1008.95 | 984.21 | 983.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 972.85 | 986.67 | 984.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-15 10:15:00 | 1006.20 | 986.63 | 984.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-09-29 11:15:00 | 990.50 | 997.76 | 991.80 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 09:15:00 | 943.50 | 987.16 | 987.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 11:15:00 | 940.75 | 986.26 | 986.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 938.05 | 931.05 | 951.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 11:15:00 | 928.10 | 931.04 | 951.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 942.50 | 931.73 | 950.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-09 13:15:00 | 940.20 | 932.14 | 950.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 942.50 | 933.16 | 949.42 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-15 10:15:00 | 940.80 | 933.23 | 949.38 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-15 14:15:00 | 950.45 | 933.66 | 949.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 1025.00 | 956.82 | 956.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1111.35 | 959.00 | 957.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1801.20 | 1864.74 | 1731.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-08 09:15:00 | 1953.90 | 1867.77 | 1789.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1841.00 | 1877.75 | 1804.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-18 14:15:00 | 1762.45 | 1866.73 | 1805.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 1775.20 | 1815.78 | 1815.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 11:15:00 | 1768.30 | 1813.25 | 1814.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1788.15 | 1774.60 | 1790.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-23 12:15:00 | 1735.45 | 1774.16 | 1790.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-25 14:15:00 | 1821.15 | 1768.51 | 1786.32 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 1881.90 | 1798.99 | 1798.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1901.20 | 1800.74 | 1799.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 1833.90 | 1836.78 | 1820.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-30 14:15:00 | 1859.05 | 1836.01 | 1821.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1873.00 | 1855.17 | 1834.11 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-10 10:15:00 | 1886.00 | 1855.67 | 1835.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1841.25 | 1856.13 | 1836.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-11 10:15:00 | 1859.45 | 1856.17 | 1836.15 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-11 14:15:00 | 1812.50 | 1855.66 | 1836.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 1746.60 | 1857.41 | 1857.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1743.20 | 1852.14 | 1854.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 888.45 | 882.93 | 981.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-09 09:15:00 | 853.45 | 909.33 | 954.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 952.85 | 907.82 | 949.00 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1017.75 | 953.73 | 953.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1021.85 | 965.58 | 959.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 989.00 | 995.93 | 979.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 1000.10 | 985.34 | 977.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 978.90 | 985.33 | 977.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-26 12:15:00 | 986.90 | 985.33 | 977.79 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-08 11:15:00 | 984.30 | 995.24 | 985.42 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 910.05 | 991.39 | 991.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1148.60 | 974.95 | 974.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 1158.80 | 976.78 | 975.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1029.99 | 1010.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-14 14:15:00 | 1036.30 | 1030.01 | 1010.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.41 | 1017.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.64 | 1017.07 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 1000.30 | 1033.97 | 1034.02 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 1047.70 | 1034.13 | 1034.07 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1033.96 | 1033.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1033.83 | 1033.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.49 | 1028.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 1012.30 | 1025.14 | 1028.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 966.60 | 925.79 | 964.82 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 1095.45 | 934.55 | 933.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.85 | 935.49 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-15 10:15:00 | 1006.20 | 2023-09-29 11:15:00 | 990.50 | EXIT_EMA400 | -15.70 |
| SELL | 2023-11-07 11:15:00 | 928.10 | 2023-11-15 14:15:00 | 950.45 | EXIT_EMA400 | -22.35 |
| SELL | 2023-11-09 13:15:00 | 940.20 | 2023-11-15 14:15:00 | 950.45 | EXIT_EMA400 | -10.25 |
| SELL | 2023-11-15 10:15:00 | 940.80 | 2023-11-15 14:15:00 | 950.45 | EXIT_EMA400 | -9.65 |
| BUY | 2024-04-08 09:15:00 | 1953.90 | 2024-04-18 14:15:00 | 1762.45 | EXIT_EMA400 | -191.45 |
| SELL | 2024-07-23 12:15:00 | 1735.45 | 2024-07-25 14:15:00 | 1821.15 | EXIT_EMA400 | -85.70 |
| BUY | 2024-08-30 14:15:00 | 1859.05 | 2024-09-11 14:15:00 | 1812.50 | EXIT_EMA400 | -46.55 |
| BUY | 2024-09-10 10:15:00 | 1886.00 | 2024-09-11 14:15:00 | 1812.50 | EXIT_EMA400 | -73.50 |
| BUY | 2024-09-11 10:15:00 | 1859.45 | 2024-09-11 14:15:00 | 1812.50 | EXIT_EMA400 | -46.95 |
| SELL | 2025-04-09 09:15:00 | 853.45 | 2025-04-16 09:15:00 | 952.85 | EXIT_EMA400 | -99.40 |
| BUY | 2025-06-26 12:15:00 | 986.90 | 2025-06-27 09:15:00 | 1014.22 | TARGET | 27.32 |
| BUY | 2025-06-24 11:15:00 | 1000.10 | 2025-07-08 11:15:00 | 984.30 | EXIT_EMA400 | -15.80 |
| BUY | 2025-10-14 14:15:00 | 1036.30 | 2025-10-27 15:15:00 | 1017.00 | EXIT_EMA400 | -19.30 |
| SELL | 2026-01-08 09:15:00 | 1012.30 | 2026-01-09 12:15:00 | 964.47 | TARGET | 47.83 |
