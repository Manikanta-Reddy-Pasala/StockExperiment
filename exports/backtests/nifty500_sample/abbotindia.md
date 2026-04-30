# Abbott India Ltd. (ABBOTINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 25435.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Total realized P&L (per unit):** 2862.70
- **Avg P&L per closed trade:** 408.96

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-06-20 09:15:00 | ALERT2 | BUY | 26846.15 | 26866.88 | 26710.37 | EMA200 retest candle locked |
| 2024-06-21 09:15:00 | ENTRY1 | BUY | 27331.20 | 26871.02 | 26717.81 | Buy entry 1 (retest1 break) |
| 2024-06-24 09:15:00 | EXIT | BUY | 26710.00 | 26879.77 | 26727.60 | Close below EMA400 |
| 2024-11-18 11:15:00 | CROSSOVER | SELL | 27457.85 | 28602.80 | 28603.60 | EMA200 below EMA400 |
| 2024-11-18 13:15:00 | ALERT1 | SELL | 27317.05 | 28577.84 | 28591.05 | Break + close below crossover candle low |
| 2024-12-02 11:15:00 | ALERT2 | SELL | 28150.60 | 28094.40 | 28305.98 | EMA200 retest candle locked |
| 2024-12-13 15:15:00 | CROSSOVER | BUY | 28648.75 | 28453.44 | 28452.94 | EMA200 above EMA400 |
| 2024-12-16 14:15:00 | CROSSOVER | SELL | 28243.75 | 28450.42 | 28451.45 | EMA200 below EMA400 |
| 2024-12-17 09:15:00 | ALERT1 | SELL | 28152.00 | 28445.49 | 28448.96 | Break + close below crossover candle low |
| 2024-12-19 10:15:00 | ALERT2 | SELL | 28572.50 | 28408.13 | 28429.25 | EMA200 retest candle locked |
| 2024-12-20 11:15:00 | CROSSOVER | BUY | 29016.95 | 28451.65 | 28450.59 | EMA200 above EMA400 |
| 2024-12-27 09:15:00 | ALERT1 | BUY | 29488.95 | 28490.79 | 28471.48 | Break + close above crossover candle high |
| 2025-01-10 11:15:00 | ALERT2 | BUY | 29122.05 | 29140.78 | 28860.65 | EMA200 retest candle locked |
| 2025-01-21 13:15:00 | CROSSOVER | SELL | 27773.70 | 28653.83 | 28654.53 | EMA200 below EMA400 |
| 2025-01-21 14:15:00 | ALERT1 | SELL | 27659.50 | 28643.94 | 28649.57 | Break + close below crossover candle low |
| 2025-02-05 13:15:00 | ALERT2 | SELL | 27530.00 | 27470.35 | 27966.73 | EMA200 retest candle locked |
| 2025-02-21 09:15:00 | CROSSOVER | BUY | 29006.75 | 28298.56 | 28298.00 | EMA200 above EMA400 |
| 2025-02-24 13:15:00 | ALERT1 | BUY | 29448.00 | 28368.29 | 28333.75 | Break + close above crossover candle high |
| 2025-03-13 14:15:00 | ALERT2 | BUY | 29601.50 | 29656.49 | 29127.28 | EMA200 retest candle locked |
| 2025-03-17 11:15:00 | ENTRY1 | BUY | 29829.05 | 29657.88 | 29138.46 | Buy entry 1 (retest1 break) |
| 2025-04-02 14:15:00 | EXIT | BUY | 29560.00 | 30121.34 | 29583.98 | Close below EMA400 |
| 2025-09-05 11:15:00 | CROSSOVER | SELL | 31030.00 | 32680.45 | 32688.08 | EMA200 below EMA400 |
| 2025-09-08 13:15:00 | ALERT1 | SELL | 30945.00 | 32555.52 | 32623.86 | Break + close below crossover candle low |
| 2025-11-25 13:15:00 | ALERT2 | SELL | 29845.00 | 29694.59 | 30210.36 | EMA200 retest candle locked |
| 2025-11-27 10:15:00 | ENTRY1 | SELL | 29610.00 | 29723.74 | 30197.81 | Sell entry 1 (retest1 break) |
| 2025-11-28 15:15:00 | ALERT3 | SELL | 30010.00 | 29707.75 | 30161.83 | EMA400 retest candle locked |
| 2025-12-01 09:15:00 | ENTRY2 | SELL | 29660.00 | 29707.28 | 30159.33 | Sell entry 2 (retest2 break) |
| 2025-12-31 14:15:00 | ALERT3 | SELL | 28950.00 | 28673.02 | 29219.15 | EMA400 retest candle locked |
| 2026-01-01 11:15:00 | ENTRY2 | SELL | 28640.00 | 28677.35 | 29210.54 | Sell entry 2 (retest2 break) |
| 2026-03-04 13:15:00 | EXIT | SELL | 27555.00 | 26932.29 | 27539.10 | Close above EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2023-10-04 12:15:00 | 22739.90 | 2023-10-06 11:15:00 | 23092.95 | -353.05 |
| BUY | 2024-03-14 09:15:00 | 28046.05 | 2024-03-22 13:15:00 | 26907.05 | -1139.00 |
| BUY | 2024-06-21 09:15:00 | 27331.20 | 2024-06-24 09:15:00 | 26710.00 | -621.20 |
| BUY | 2025-03-17 11:15:00 | 29829.05 | 2025-04-02 14:15:00 | 29560.00 | -269.05 |
| SELL | 2025-11-27 10:15:00 | 29610.00 | 2026-03-04 13:15:00 | 27555.00 | 2055.00 |
| SELL | 2025-12-01 09:15:00 | 29660.00 | 2026-03-04 13:15:00 | 27555.00 | 2105.00 |
| SELL | 2026-01-01 11:15:00 | 28640.00 | 2026-03-04 13:15:00 | 27555.00 | 1085.00 |
