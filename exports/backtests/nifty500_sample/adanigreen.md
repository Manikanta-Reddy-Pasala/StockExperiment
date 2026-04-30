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
- **Winners / losers:** 1 / 13
- **Total realized P&L (per unit):** -593.50
- **Avg P&L per closed trade:** -42.39

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-09-11 10:15:00 | ENTRY2 | BUY | 1859.45 | 1856.17 | 1836.15 | Buy entry 2 (retest2 break) |
| 2024-09-11 14:15:00 | EXIT | BUY | 1812.50 | 1855.66 | 1836.29 | Close below EMA400 |
| 2024-10-16 11:15:00 | CROSSOVER | SELL | 1746.60 | 1857.41 | 1857.44 | EMA200 below EMA400 |
| 2024-10-17 09:15:00 | ALERT1 | SELL | 1743.20 | 1852.14 | 1854.77 | Break + close below crossover candle low |
| 2025-03-13 09:15:00 | ALERT2 | SELL | 888.45 | 882.93 | 981.01 | EMA200 retest candle locked |
| 2025-04-09 09:15:00 | ENTRY1 | SELL | 853.45 | 909.33 | 954.38 | Sell entry 1 (retest1 break) |
| 2025-04-16 09:15:00 | EXIT | SELL | 952.85 | 907.82 | 949.00 | Close above EMA400 |
| 2025-05-26 10:15:00 | CROSSOVER | BUY | 1017.75 | 953.73 | 953.63 | EMA200 above EMA400 |
| 2025-05-29 12:15:00 | ALERT1 | BUY | 1021.85 | 965.58 | 959.92 | Break + close above crossover candle high |
| 2025-06-13 12:15:00 | ALERT2 | BUY | 989.00 | 995.93 | 979.49 | EMA200 retest candle locked |
| 2025-06-24 11:15:00 | ENTRY1 | BUY | 1000.10 | 985.34 | 977.21 | Buy entry 1 (retest1 break) |
| 2025-06-26 10:15:00 | ALERT3 | BUY | 978.90 | 985.33 | 977.72 | EMA400 retest candle locked |
| 2025-06-26 12:15:00 | ENTRY2 | BUY | 986.90 | 985.33 | 977.79 | Buy entry 2 (retest2 break) |
| 2025-07-08 11:15:00 | EXIT | BUY | 984.30 | 995.24 | 985.42 | Close below EMA400 |
| 2025-08-08 15:15:00 | CROSSOVER | SELL | 910.05 | 991.39 | 991.66 | EMA200 below EMA400 |
| 2025-09-22 13:15:00 | CROSSOVER | BUY | 1148.60 | 974.95 | 974.54 | EMA200 above EMA400 |
| 2025-09-22 14:15:00 | ALERT1 | BUY | 1158.80 | 976.78 | 975.46 | Break + close above crossover candle high |
| 2025-10-14 11:15:00 | ALERT2 | BUY | 1027.80 | 1029.99 | 1010.09 | EMA200 retest candle locked |
| 2025-10-14 14:15:00 | ENTRY1 | BUY | 1036.30 | 1030.01 | 1010.39 | Buy entry 1 (retest1 break) |
| 2025-10-27 10:15:00 | ALERT3 | BUY | 1020.80 | 1034.41 | 1017.02 | EMA400 retest candle locked |
| 2025-10-27 15:15:00 | EXIT | BUY | 1017.00 | 1033.64 | 1017.07 | Close below EMA400 |
| 2025-12-11 10:15:00 | CROSSOVER | SELL | 1000.30 | 1033.97 | 1034.02 | EMA200 below EMA400 |
| 2025-12-15 13:15:00 | CROSSOVER | BUY | 1047.70 | 1034.13 | 1034.07 | EMA200 above EMA400 |
| 2025-12-16 15:15:00 | CROSSOVER | SELL | 1024.00 | 1033.96 | 1033.99 | EMA200 below EMA400 |
| 2025-12-17 09:15:00 | ALERT1 | SELL | 1021.60 | 1033.83 | 1033.93 | Break + close below crossover candle low |
| 2026-01-01 09:15:00 | ALERT2 | SELL | 1033.20 | 1024.49 | 1028.49 | EMA200 retest candle locked |
| 2026-01-08 09:15:00 | ENTRY1 | SELL | 1012.30 | 1025.14 | 1028.24 | Sell entry 1 (retest1 break) |
| 2026-02-04 11:15:00 | EXIT | SELL | 966.60 | 925.79 | 964.82 | Close above EMA400 |
| 2026-04-15 14:15:00 | CROSSOVER | BUY | 1095.45 | 934.55 | 933.83 | EMA200 above EMA400 |
| 2026-04-16 09:15:00 | ALERT1 | BUY | 1105.80 | 937.85 | 935.49 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2023-09-15 10:15:00 | 1006.20 | 2023-09-29 11:15:00 | 990.50 | -15.70 |
| SELL | 2023-11-07 11:15:00 | 928.10 | 2023-11-15 14:15:00 | 950.45 | -22.35 |
| SELL | 2023-11-09 13:15:00 | 940.20 | 2023-11-15 14:15:00 | 950.45 | -10.25 |
| SELL | 2023-11-15 10:15:00 | 940.80 | 2023-11-15 14:15:00 | 950.45 | -9.65 |
| BUY | 2024-04-08 09:15:00 | 1953.90 | 2024-04-18 14:15:00 | 1762.45 | -191.45 |
| SELL | 2024-07-23 12:15:00 | 1735.45 | 2024-07-25 14:15:00 | 1821.15 | -85.70 |
| BUY | 2024-08-30 14:15:00 | 1859.05 | 2024-09-11 14:15:00 | 1812.50 | -46.55 |
| BUY | 2024-09-10 10:15:00 | 1886.00 | 2024-09-11 14:15:00 | 1812.50 | -73.50 |
| BUY | 2024-09-11 10:15:00 | 1859.45 | 2024-09-11 14:15:00 | 1812.50 | -46.95 |
| SELL | 2025-04-09 09:15:00 | 853.45 | 2025-04-16 09:15:00 | 952.85 | -99.40 |
| BUY | 2025-06-24 11:15:00 | 1000.10 | 2025-07-08 11:15:00 | 984.30 | -15.80 |
| BUY | 2025-06-26 12:15:00 | 986.90 | 2025-07-08 11:15:00 | 984.30 | -2.60 |
| BUY | 2025-10-14 14:15:00 | 1036.30 | 2025-10-27 15:15:00 | 1017.00 | -19.30 |
| SELL | 2026-01-08 09:15:00 | 1012.30 | 2026-02-04 11:15:00 | 966.60 | 45.70 |
