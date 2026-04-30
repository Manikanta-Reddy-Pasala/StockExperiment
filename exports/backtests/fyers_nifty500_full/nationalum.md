# National Aluminium Co. Ltd. (NATIONALUM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 395.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 26.72
- **Avg P&L per closed trade:** 5.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 175.19 | 186.09 | 186.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 174.11 | 185.97 | 186.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 178.78 | 178.71 | 181.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-04 09:15:00 | 175.34 | 179.76 | 181.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-13 09:15:00 | 183.70 | 177.76 | 180.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 195.62 | 181.97 | 181.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 11:15:00 | 198.47 | 182.14 | 182.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 221.49 | 223.09 | 211.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-18 09:15:00 | 237.82 | 223.06 | 212.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-13 09:15:00 | 226.99 | 239.75 | 228.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 09:15:00 | 206.25 | 222.87 | 222.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 200.88 | 220.45 | 221.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 09:15:00 | 196.23 | 195.15 | 202.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-24 13:15:00 | 189.76 | 195.35 | 202.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 171.85 | 163.16 | 171.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 12:15:00 | 172.22 | 163.40 | 171.85 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 185.45 | 176.39 | 176.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 188.00 | 176.51 | 176.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 181.50 | 181.64 | 179.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 183.26 | 181.65 | 179.51 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 188.93 | 190.12 | 186.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 14:15:00 | 189.73 | 190.07 | 186.74 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 185.83 | 189.86 | 186.78 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-04 09:15:00 | 175.34 | 2024-09-13 09:15:00 | 183.70 | EXIT_EMA400 | -8.36 |
| BUY | 2024-11-18 09:15:00 | 237.82 | 2024-12-13 09:15:00 | 226.99 | EXIT_EMA400 | -10.83 |
| SELL | 2025-02-24 13:15:00 | 189.76 | 2025-04-07 09:15:00 | 151.21 | TARGET | 38.55 |
| BUY | 2025-06-20 09:15:00 | 183.26 | 2025-06-27 09:15:00 | 194.52 | TARGET | 11.26 |
| BUY | 2025-07-29 14:15:00 | 189.73 | 2025-07-31 09:15:00 | 185.83 | EXIT_EMA400 | -3.90 |
