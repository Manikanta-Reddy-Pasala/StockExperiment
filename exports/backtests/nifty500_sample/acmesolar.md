# ACME Solar Holdings Ltd. (ACMESOLAR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-13 09:15:00 → 2026-04-30 15:30:00 (2505 bars)
- **Last close:** 302.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Total realized P&L (per unit):** 37.00
- **Avg P&L per closed trade:** 18.50

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2025-05-09 11:15:00 | CROSSOVER | BUY | 212.33 | 209.04 | 209.03 | EMA200 above EMA400 |
| 2025-05-09 14:15:00 | ALERT1 | BUY | 212.68 | 209.12 | 209.07 | Break + close above crossover candle high |
| 2025-06-13 12:15:00 | ALERT2 | BUY | 241.90 | 241.93 | 231.48 | EMA200 retest candle locked |
| 2025-06-13 14:15:00 | ENTRY1 | BUY | 245.20 | 241.95 | 231.59 | Buy entry 1 (retest1 break) |
| 2025-09-23 14:15:00 | ALERT3 | BUY | 288.45 | 295.25 | 284.25 | EMA400 retest candle locked |
| 2025-09-24 14:15:00 | EXIT | BUY | 283.75 | 294.73 | 284.37 | Close below EMA400 |
| 2025-11-07 13:15:00 | CROSSOVER | SELL | 265.75 | 281.78 | 281.85 | EMA200 below EMA400 |
| 2025-11-07 14:15:00 | ALERT1 | SELL | 264.85 | 281.61 | 281.76 | Break + close below crossover candle low |
| 2025-12-22 12:15:00 | ALERT2 | SELL | 235.64 | 235.64 | 248.54 | EMA200 retest candle locked |
| 2025-12-26 13:15:00 | ENTRY1 | SELL | 231.70 | 235.77 | 247.27 | Sell entry 1 (retest1 break) |
| 2026-02-10 09:15:00 | EXIT | SELL | 233.25 | 222.31 | 230.47 | Close above EMA400 |
| 2026-03-17 13:15:00 | CROSSOVER | BUY | 247.40 | 231.91 | 231.87 | EMA200 above EMA400 |
| 2026-03-18 09:15:00 | ALERT1 | BUY | 249.26 | 232.38 | 232.11 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2025-06-13 14:15:00 | 245.20 | 2025-09-24 14:15:00 | 283.75 | 38.55 |
| SELL | 2025-12-26 13:15:00 | 231.70 | 2026-02-10 09:15:00 | 233.25 | -1.55 |
