# Aditya Birla Real Estate Ltd. (ABREL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-10-10 09:15:00 → 2026-04-30 15:30:00 (2666 bars)
- **Last close:** 1486.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 1 / 0
- **Total realized P&L (per unit):** 46.50
- **Avg P&L per closed trade:** 46.50

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2025-05-26 14:15:00 | CROSSOVER | BUY | 2196.20 | 2002.01 | 2001.69 | EMA200 above EMA400 |
| 2025-05-27 09:15:00 | ALERT1 | BUY | 2232.30 | 2006.20 | 2003.80 | Break + close above crossover candle high |
| 2025-07-04 11:15:00 | ALERT2 | BUY | 2315.70 | 2333.36 | 2236.41 | EMA200 retest candle locked |
| 2025-07-28 14:15:00 | CROSSOVER | SELL | 1982.40 | 2198.93 | 2198.94 | EMA200 below EMA400 |
| 2025-07-28 15:15:00 | ALERT1 | SELL | 1952.00 | 2196.47 | 2197.71 | Break + close below crossover candle low |
| 2025-09-15 14:15:00 | ALERT2 | SELL | 1865.00 | 1852.21 | 1940.99 | EMA200 retest candle locked |
| 2025-09-24 10:15:00 | ENTRY1 | SELL | 1843.10 | 1869.17 | 1932.81 | Sell entry 1 (retest1 break) |
| 2025-10-29 12:15:00 | ALERT3 | SELL | 1781.60 | 1708.25 | 1787.15 | EMA400 retest candle locked |
| 2025-10-29 13:15:00 | EXIT | SELL | 1796.60 | 1709.13 | 1787.20 | Close above EMA400 |
| 2026-04-30 10:15:00 | CROSSOVER | BUY | 1487.10 | 1348.44 | 1347.92 | EMA200 above EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2025-09-24 10:15:00 | 1843.10 | 2025-10-29 13:15:00 | 1796.60 | 46.50 |
