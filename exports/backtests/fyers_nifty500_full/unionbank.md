# Union Bank of India (UNIONBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 165.94
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 7 |
| EXIT | 5 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 7 / 5
- **Target hits / EMA400 exits:** 7 / 5
- **Total realized P&L (per unit):** 12.29
- **Avg P&L per closed trade:** 1.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 129.13 | 120.19 | 120.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 15:15:00 | 129.50 | 120.86 | 120.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 13:15:00 | 123.07 | 123.33 | 121.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 125.36 | 121.51 | 121.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 116.25 | 121.63 | 121.38 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 10:15:00 | 115.30 | 121.13 | 121.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 14:15:00 | 112.67 | 120.85 | 120.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 113.01 | 112.67 | 115.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 110.00 | 112.77 | 115.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 118.73 | 112.70 | 115.17 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 122.93 | 115.19 | 115.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 125.35 | 115.36 | 115.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 115.76 | 120.02 | 117.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 119.08 | 119.84 | 117.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 119.08 | 119.84 | 117.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 11:15:00 | 119.66 | 119.82 | 117.94 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-09 09:15:00 | 117.84 | 119.83 | 117.99 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 128.68 | 139.96 | 140.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 127.21 | 139.83 | 139.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 137.43 | 137.12 | 138.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 12:15:00 | 136.81 | 137.16 | 138.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 137.51 | 137.16 | 138.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-22 12:15:00 | 136.05 | 137.14 | 138.18 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 134.11 | 132.28 | 134.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 136.74 | 132.70 | 134.74 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 138.46 | 136.02 | 136.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 139.80 | 136.10 | 136.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-20 09:15:00 | 139.55 | 136.92 | 136.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 148.39 | 150.82 | 146.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 149.97 | 150.74 | 147.01 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 147.20 | 150.59 | 147.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-11 09:15:00 | 149.42 | 150.58 | 147.14 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 149.42 | 151.42 | 148.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-30 10:15:00 | 150.15 | 151.34 | 148.82 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 177.84 | 186.30 | 176.77 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-10 12:15:00 | 184.83 | 185.71 | 176.93 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 177.25 | 185.41 | 177.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-12 12:15:00 | 179.74 | 185.23 | 177.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-13 09:15:00 | 176.79 | 185.06 | 177.35 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-03 09:15:00 | 125.36 | 2025-01-06 09:15:00 | 116.25 | EXIT_EMA400 | -9.11 |
| SELL | 2025-02-03 09:15:00 | 110.00 | 2025-02-05 09:15:00 | 118.73 | EXIT_EMA400 | -8.73 |
| BUY | 2025-04-07 15:15:00 | 119.08 | 2025-04-08 09:15:00 | 122.56 | TARGET | 3.48 |
| BUY | 2025-04-08 11:15:00 | 119.66 | 2025-04-09 09:15:00 | 117.84 | EXIT_EMA400 | -1.82 |
| SELL | 2025-08-21 12:15:00 | 136.81 | 2025-08-26 09:15:00 | 132.53 | TARGET | 4.28 |
| SELL | 2025-08-22 12:15:00 | 136.05 | 2025-08-26 12:15:00 | 129.65 | TARGET | 6.40 |
| BUY | 2025-10-20 09:15:00 | 139.55 | 2025-10-30 12:15:00 | 148.62 | TARGET | 9.07 |
| BUY | 2025-12-30 10:15:00 | 150.15 | 2025-12-31 10:15:00 | 154.13 | TARGET | 3.98 |
| BUY | 2025-12-11 09:15:00 | 149.42 | 2026-01-02 11:15:00 | 156.26 | TARGET | 6.84 |
| BUY | 2025-12-09 11:15:00 | 149.97 | 2026-01-05 09:15:00 | 158.86 | TARGET | 8.89 |
| BUY | 2026-03-10 12:15:00 | 184.83 | 2026-03-13 09:15:00 | 176.79 | EXIT_EMA400 | -8.04 |
| BUY | 2026-03-12 12:15:00 | 179.74 | 2026-03-13 09:15:00 | 176.79 | EXIT_EMA400 | -2.95 |
