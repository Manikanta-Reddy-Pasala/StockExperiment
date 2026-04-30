# Aarti Industries Ltd. (AARTIIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 507.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Total realized P&L (per unit):** -31.60
- **Avg P&L per closed trade:** -3.95

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-06-10 12:15:00 | ALERT2 | SELL | 654.60 | 648.94 | 659.86 | EMA200 retest candle locked |
| 2024-06-24 13:15:00 | CROSSOVER | BUY | 709.35 | 666.83 | 666.81 | EMA200 above EMA400 |
| 2024-07-02 09:15:00 | ALERT1 | BUY | 718.00 | 675.64 | 671.63 | Break + close above crossover candle high |
| 2024-07-10 09:15:00 | ALERT2 | BUY | 687.75 | 688.33 | 679.44 | EMA200 retest candle locked |
| 2024-07-12 10:15:00 | ENTRY1 | BUY | 707.65 | 689.77 | 680.83 | Buy entry 1 (retest1 break) |
| 2024-07-19 10:15:00 | ALERT3 | BUY | 685.10 | 693.28 | 683.90 | EMA400 retest candle locked |
| 2024-07-19 12:15:00 | EXIT | BUY | 680.00 | 693.06 | 683.88 | Close below EMA400 |
| 2024-08-20 13:15:00 | CROSSOVER | SELL | 622.30 | 688.47 | 688.77 | EMA200 below EMA400 |
| 2024-08-23 13:15:00 | ALERT1 | SELL | 618.70 | 676.06 | 682.21 | Break + close below crossover candle low |
| 2025-01-10 12:15:00 | ALERT2 | SELL | 425.80 | 425.31 | 456.18 | EMA200 retest candle locked |
| 2025-01-13 15:15:00 | ENTRY1 | SELL | 408.00 | 424.84 | 454.44 | Sell entry 1 (retest1 break) |
| 2025-01-21 09:15:00 | EXIT | SELL | 453.70 | 427.68 | 451.00 | Close above EMA400 |
| 2025-05-07 09:15:00 | CROSSOVER | BUY | 440.85 | 414.91 | 414.79 | EMA200 above EMA400 |
| 2025-05-07 10:15:00 | ALERT1 | BUY | 448.25 | 415.24 | 414.96 | Break + close above crossover candle high |
| 2025-06-13 10:15:00 | ALERT2 | BUY | 464.80 | 466.02 | 450.95 | EMA200 retest candle locked |
| 2025-06-25 09:15:00 | ENTRY1 | BUY | 468.00 | 459.77 | 451.08 | Buy entry 1 (retest1 break) |
| 2025-07-08 10:15:00 | EXIT | BUY | 456.55 | 467.70 | 458.08 | Close below EMA400 |
| 2025-07-24 15:15:00 | CROSSOVER | SELL | 433.55 | 452.93 | 452.95 | EMA200 below EMA400 |
| 2025-07-25 09:15:00 | ALERT1 | SELL | 429.80 | 452.70 | 452.83 | Break + close below crossover candle low |
| 2025-07-30 13:15:00 | ALERT2 | SELL | 450.00 | 449.29 | 450.98 | EMA200 retest candle locked |
| 2025-07-30 14:15:00 | ENTRY1 | SELL | 445.30 | 449.25 | 450.95 | Sell entry 1 (retest1 break) |
| 2025-11-03 09:15:00 | ALERT3 | SELL | 388.20 | 381.26 | 389.32 | EMA400 retest candle locked |
| 2025-11-03 11:15:00 | EXIT | SELL | 389.40 | 381.41 | 389.32 | Close above EMA400 |
| 2026-02-05 13:15:00 | CROSSOVER | BUY | 447.45 | 375.13 | 374.83 | EMA200 above EMA400 |
| 2026-02-05 14:15:00 | ALERT1 | BUY | 454.90 | 375.93 | 375.23 | Break + close above crossover candle high |
| 2026-03-04 09:15:00 | ALERT2 | BUY | 417.50 | 428.97 | 410.26 | EMA200 retest candle locked |
| 2026-03-10 14:15:00 | ENTRY1 | BUY | 427.90 | 425.11 | 411.01 | Buy entry 1 (retest1 break) |
| 2026-03-16 09:15:00 | ALERT3 | BUY | 413.90 | 426.87 | 413.48 | EMA400 retest candle locked |
| 2026-03-16 11:15:00 | ENTRY2 | BUY | 424.35 | 426.74 | 413.55 | Buy entry 2 (retest2 break) |
| 2026-03-19 13:15:00 | EXIT | BUY | 414.35 | 426.31 | 414.75 | Close below EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2023-09-26 09:15:00 | 500.85 | 2023-09-28 12:15:00 | 490.40 | -10.45 |
| BUY | 2024-01-09 12:15:00 | 606.30 | 2024-03-13 09:15:00 | 637.60 | 31.30 |
| BUY | 2024-07-12 10:15:00 | 707.65 | 2024-07-19 12:15:00 | 680.00 | -27.65 |
| SELL | 2025-01-13 15:15:00 | 408.00 | 2025-01-21 09:15:00 | 453.70 | -45.70 |
| BUY | 2025-06-25 09:15:00 | 468.00 | 2025-07-08 10:15:00 | 456.55 | -11.45 |
| SELL | 2025-07-30 14:15:00 | 445.30 | 2025-11-03 11:15:00 | 389.40 | 55.90 |
| BUY | 2026-03-10 14:15:00 | 427.90 | 2026-03-19 13:15:00 | 414.35 | -13.55 |
| BUY | 2026-03-16 11:15:00 | 424.35 | 2026-03-19 13:15:00 | 414.35 | -10.00 |
