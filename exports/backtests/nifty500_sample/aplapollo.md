# APL Apollo Tubes Ltd. (APLAPOLLO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1905.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 10 |
| ENTRY2 | 3 |
| EXIT | 10 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Total realized P&L (per unit):** -423.50
- **Avg P&L per closed trade:** -32.58

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2025-03-13 15:15:00 | ENTRY1 | SELL | 1372.80 | 1439.50 | 1463.25 | Sell entry 1 (retest1 break) |
| 2025-03-19 09:15:00 | EXIT | SELL | 1462.85 | 1437.58 | 1460.52 | Close above EMA400 |
| 2025-04-02 13:15:00 | CROSSOVER | BUY | 1551.60 | 1476.10 | 1476.06 | EMA200 above EMA400 |
| 2025-04-02 15:15:00 | ALERT1 | BUY | 1562.20 | 1477.79 | 1476.91 | Break + close above crossover candle high |
| 2025-04-07 09:15:00 | ALERT2 | BUY | 1431.00 | 1486.15 | 1481.36 | EMA200 retest candle locked |
| 2025-04-08 12:15:00 | ENTRY1 | BUY | 1485.90 | 1481.96 | 1479.43 | Buy entry 1 (retest1 break) |
| 2025-04-08 12:15:00 | ALERT3 | BUY | 1485.90 | 1481.96 | 1479.43 | EMA400 retest candle locked |
| 2025-04-08 13:15:00 | EXIT | BUY | 1475.40 | 1481.89 | 1479.41 | Close below EMA400 |
| 2025-07-25 14:15:00 | CROSSOVER | SELL | 1545.10 | 1717.63 | 1717.64 | EMA200 below EMA400 |
| 2025-07-28 09:15:00 | ALERT1 | SELL | 1497.50 | 1713.79 | 1715.71 | Break + close below crossover candle low |
| 2025-08-20 09:15:00 | ALERT2 | SELL | 1638.10 | 1630.45 | 1660.50 | EMA200 retest candle locked |
| 2025-08-28 11:15:00 | ENTRY1 | SELL | 1608.60 | 1633.29 | 1657.01 | Sell entry 1 (retest1 break) |
| 2025-09-02 10:15:00 | EXIT | SELL | 1661.50 | 1631.28 | 1653.65 | Close above EMA400 |
| 2025-09-23 09:15:00 | CROSSOVER | BUY | 1669.70 | 1665.95 | 1665.95 | EMA200 above EMA400 |
| 2025-09-25 09:15:00 | ALERT1 | BUY | 1689.40 | 1667.84 | 1666.92 | Break + close above crossover candle high |
| 2025-09-26 13:15:00 | ALERT2 | BUY | 1667.90 | 1669.37 | 1667.76 | EMA200 retest candle locked |
| 2025-09-29 09:15:00 | ENTRY1 | BUY | 1685.10 | 1669.38 | 1667.79 | Buy entry 1 (retest1 break) |
| 2025-11-19 13:15:00 | EXIT | BUY | 1730.00 | 1761.58 | 1735.76 | Close below EMA400 |
| 2026-04-06 13:15:00 | CROSSOVER | SELL | 1913.00 | 2016.66 | 2016.79 | EMA200 below EMA400 |
| 2026-04-07 09:15:00 | ALERT1 | SELL | 1880.20 | 2013.35 | 2015.12 | Break + close below crossover candle low |
| 2026-04-08 12:15:00 | ALERT2 | SELL | 2033.90 | 2005.84 | 2011.15 | EMA200 retest candle locked |
| 2026-04-13 09:15:00 | ENTRY1 | SELL | 1999.40 | 2011.73 | 2013.77 | Sell entry 1 (retest1 break) |
| 2026-04-13 09:15:00 | ALERT3 | SELL | 1999.40 | 2011.73 | 2013.77 | EMA400 retest candle locked |
| 2026-04-13 14:15:00 | ENTRY2 | SELL | 1980.40 | 2011.26 | 2013.48 | Sell entry 2 (retest2 break) |
| 2026-04-15 09:15:00 | ALERT3 | SELL | 2004.30 | 2010.88 | 2013.27 | EMA400 retest candle locked |
| 2026-04-15 10:15:00 | EXIT | SELL | 2019.80 | 2010.97 | 2013.30 | Close above EMA400 |
| 2026-04-17 11:15:00 | CROSSOVER | BUY | 2085.90 | 2015.81 | 2015.63 | EMA200 above EMA400 |
| 2026-04-17 13:15:00 | ALERT1 | BUY | 2106.90 | 2017.37 | 2016.41 | Break + close above crossover candle high |
| 2026-04-23 14:15:00 | ALERT2 | BUY | 2023.10 | 2035.67 | 2026.34 | EMA200 retest candle locked |
| 2026-04-30 13:15:00 | CROSSOVER | SELL | 1902.30 | 2018.17 | 2018.52 | EMA200 below EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2024-01-15 13:15:00 | 1538.05 | 2024-02-29 12:15:00 | 1497.95 | 40.10 |
| BUY | 2024-06-07 14:15:00 | 1621.95 | 2024-06-13 10:15:00 | 1571.95 | -50.00 |
| SELL | 2024-09-06 14:15:00 | 1400.15 | 2024-09-13 12:15:00 | 1478.95 | -78.80 |
| SELL | 2024-11-26 14:15:00 | 1479.50 | 2024-11-29 10:15:00 | 1518.55 | -39.05 |
| SELL | 2024-11-27 09:15:00 | 1474.00 | 2024-11-29 10:15:00 | 1518.55 | -44.55 |
| SELL | 2024-11-28 09:15:00 | 1472.15 | 2024-11-29 10:15:00 | 1518.55 | -46.40 |
| BUY | 2024-12-31 12:15:00 | 1572.70 | 2025-01-08 11:15:00 | 1536.25 | -36.45 |
| SELL | 2025-03-13 15:15:00 | 1372.80 | 2025-03-19 09:15:00 | 1462.85 | -90.05 |
| BUY | 2025-04-08 12:15:00 | 1485.90 | 2025-04-08 13:15:00 | 1475.40 | -10.50 |
| SELL | 2025-08-28 11:15:00 | 1608.60 | 2025-09-02 10:15:00 | 1661.50 | -52.90 |
| BUY | 2025-09-29 09:15:00 | 1685.10 | 2025-11-19 13:15:00 | 1730.00 | 44.90 |
| SELL | 2026-04-13 09:15:00 | 1999.40 | 2026-04-15 10:15:00 | 2019.80 | -20.40 |
| SELL | 2026-04-13 14:15:00 | 1980.40 | 2026-04-15 10:15:00 | 2019.80 | -39.40 |
