# AU Small Finance Bank Ltd. (AUBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1015.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Total realized P&L (per unit):** -210.60
- **Avg P&L per closed trade:** -23.40

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-08-28 15:15:00 | ENTRY2 | SELL | 628.30 | 632.41 | 637.01 | Sell entry 2 (retest2 break) |
| 2024-08-29 10:15:00 | ALERT3 | SELL | 635.00 | 632.44 | 636.97 | EMA400 retest candle locked |
| 2024-08-29 11:15:00 | EXIT | SELL | 648.95 | 632.60 | 637.03 | Close above EMA400 |
| 2024-09-03 13:15:00 | CROSSOVER | BUY | 683.30 | 641.23 | 641.07 | EMA200 above EMA400 |
| 2024-09-04 12:15:00 | ALERT1 | BUY | 691.90 | 643.50 | 642.22 | Break + close above crossover candle high |
| 2024-10-09 13:15:00 | ALERT2 | BUY | 707.65 | 712.97 | 690.78 | EMA200 retest candle locked |
| 2024-10-29 10:15:00 | CROSSOVER | SELL | 619.75 | 680.35 | 680.62 | EMA200 below EMA400 |
| 2024-10-29 12:15:00 | ALERT1 | SELL | 616.20 | 679.12 | 680.00 | Break + close below crossover candle low |
| 2024-12-30 14:15:00 | ALERT2 | SELL | 587.50 | 577.64 | 602.11 | EMA200 retest candle locked |
| 2025-02-14 09:15:00 | ENTRY1 | SELL | 545.05 | 581.81 | 588.60 | Sell entry 1 (retest1 break) |
| 2025-03-24 10:15:00 | ALERT3 | SELL | 552.75 | 537.88 | 554.76 | EMA400 retest candle locked |
| 2025-03-24 11:15:00 | ENTRY2 | SELL | 548.30 | 537.99 | 554.73 | Sell entry 2 (retest2 break) |
| 2025-03-25 09:15:00 | ALERT3 | SELL | 553.00 | 538.64 | 554.64 | EMA400 retest candle locked |
| 2025-03-25 10:15:00 | EXIT | SELL | 556.70 | 538.82 | 554.66 | Close above EMA400 |
| 2025-04-22 15:15:00 | CROSSOVER | BUY | 613.80 | 559.99 | 559.79 | EMA200 above EMA400 |
| 2025-04-23 09:15:00 | ALERT1 | BUY | 647.25 | 560.86 | 560.23 | Break + close above crossover candle high |
| 2025-07-17 09:15:00 | ALERT2 | BUY | 790.55 | 791.70 | 748.64 | EMA200 retest candle locked |
| 2025-09-04 10:15:00 | CROSSOVER | SELL | 703.25 | 742.56 | 742.58 | EMA200 below EMA400 |
| 2025-09-04 11:15:00 | ALERT1 | SELL | 700.30 | 742.14 | 742.37 | Break + close below crossover candle low |
| 2025-09-23 09:15:00 | ALERT2 | SELL | 729.45 | 722.30 | 729.91 | EMA200 retest candle locked |
| 2025-10-08 13:15:00 | CROSSOVER | BUY | 768.20 | 734.56 | 734.55 | EMA200 above EMA400 |
| 2025-10-14 14:15:00 | ALERT1 | BUY | 772.50 | 742.13 | 738.59 | Break + close above crossover candle high |
| 2026-01-13 11:15:00 | ALERT2 | BUY | 977.20 | 978.09 | 937.11 | EMA200 retest candle locked |
| 2026-01-16 09:15:00 | ENTRY1 | BUY | 1006.65 | 977.88 | 939.39 | Buy entry 1 (retest1 break) |
| 2026-01-27 11:15:00 | EXIT | BUY | 950.05 | 984.02 | 950.60 | Close below EMA400 |
| 2026-03-16 09:15:00 | CROSSOVER | SELL | 889.95 | 961.60 | 961.81 | EMA200 below EMA400 |
| 2026-03-16 10:15:00 | ALERT1 | SELL | 881.70 | 960.81 | 961.41 | Break + close below crossover candle low |
| 2026-04-08 09:15:00 | ALERT2 | SELL | 941.05 | 911.16 | 930.94 | EMA200 retest candle locked |
| 2026-04-21 15:15:00 | CROSSOVER | BUY | 1040.00 | 945.17 | 945.02 | EMA200 above EMA400 |
| 2026-04-22 09:15:00 | ALERT1 | BUY | 1041.80 | 946.13 | 945.51 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2023-12-20 09:15:00 | 758.35 | 2024-01-17 11:15:00 | 747.40 | -10.95 |
| SELL | 2024-04-03 10:15:00 | 579.05 | 2024-04-04 09:15:00 | 634.50 | -55.45 |
| BUY | 2024-06-05 10:15:00 | 648.15 | 2024-07-08 11:15:00 | 646.35 | -1.80 |
| BUY | 2024-06-24 10:15:00 | 671.70 | 2024-07-08 11:15:00 | 646.35 | -25.35 |
| SELL | 2024-08-23 11:15:00 | 629.20 | 2024-08-29 11:15:00 | 648.95 | -19.75 |
| SELL | 2024-08-28 15:15:00 | 628.30 | 2024-08-29 11:15:00 | 648.95 | -20.65 |
| SELL | 2025-02-14 09:15:00 | 545.05 | 2025-03-25 10:15:00 | 556.70 | -11.65 |
| SELL | 2025-03-24 11:15:00 | 548.30 | 2025-03-25 10:15:00 | 556.70 | -8.40 |
| BUY | 2026-01-16 09:15:00 | 1006.65 | 2026-01-27 11:15:00 | 950.05 | -56.60 |
