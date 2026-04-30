# Union Bank of India (UNIONBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 165.94
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 10 |
| ENTRY1 | 6 |
| ENTRY2 | 8 |
| EXIT | 6 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / EMA400 exits:** 8 / 6
- **Total realized P&L (per unit):** 18.37
- **Avg P&L per closed trade:** 1.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 138.97 | 146.56 | 146.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 138.26 | 146.40 | 146.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 140.90 | 140.84 | 143.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-16 13:15:00 | 139.66 | 140.83 | 143.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 127.40 | 123.79 | 127.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-24 10:15:00 | 126.98 | 123.90 | 127.70 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 127.02 | 124.10 | 127.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-25 10:15:00 | 126.38 | 124.12 | 127.69 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 119.38 | 115.83 | 119.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 12:15:00 | 120.06 | 115.94 | 119.73 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 128.54 | 120.43 | 120.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 129.19 | 120.77 | 120.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 13:15:00 | 123.07 | 123.32 | 122.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 125.31 | 121.51 | 121.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 116.25 | 121.63 | 121.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 09:15:00 | 114.43 | 121.18 | 121.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 14:15:00 | 112.67 | 120.84 | 121.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 113.01 | 112.67 | 115.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 110.05 | 112.69 | 115.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 118.71 | 112.63 | 115.24 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 122.93 | 115.18 | 115.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 125.35 | 115.35 | 115.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 115.67 | 120.02 | 117.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 119.17 | 119.84 | 117.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 119.17 | 119.84 | 117.94 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 10:15:00 | 119.20 | 119.82 | 117.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-09 09:15:00 | 117.84 | 119.83 | 118.01 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 128.65 | 139.95 | 139.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 127.21 | 139.82 | 139.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 137.43 | 137.12 | 138.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 12:15:00 | 136.81 | 137.16 | 138.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 134.11 | 132.27 | 134.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 136.74 | 132.69 | 134.74 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 138.46 | 136.02 | 136.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 139.89 | 136.10 | 136.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-20 09:15:00 | 139.55 | 136.93 | 136.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 148.39 | 150.82 | 146.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 149.97 | 150.74 | 147.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 148.06 | 150.59 | 147.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-11 09:15:00 | 149.42 | 150.58 | 147.14 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 149.42 | 151.42 | 148.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-30 10:15:00 | 150.14 | 151.35 | 148.82 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 177.88 | 186.23 | 176.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-10 12:15:00 | 184.83 | 185.64 | 176.74 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 177.25 | 185.33 | 177.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-12 10:15:00 | 179.20 | 185.27 | 177.07 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-13 09:15:00 | 176.81 | 184.99 | 177.17 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-07-16 13:15:00 | 139.66 | 2024-07-23 12:15:00 | 129.59 | TARGET | 10.07 |
| SELL | 2024-09-24 10:15:00 | 126.98 | 2024-09-26 09:15:00 | 124.81 | TARGET | 2.17 |
| SELL | 2024-09-25 10:15:00 | 126.38 | 2024-09-27 15:15:00 | 122.46 | TARGET | 3.92 |
| BUY | 2025-01-03 09:15:00 | 125.31 | 2025-01-06 09:15:00 | 116.25 | EXIT_EMA400 | -9.06 |
| SELL | 2025-02-03 09:15:00 | 110.05 | 2025-02-05 09:15:00 | 118.71 | EXIT_EMA400 | -8.66 |
| BUY | 2025-04-07 15:15:00 | 119.17 | 2025-04-09 09:15:00 | 117.84 | EXIT_EMA400 | -1.33 |
| BUY | 2025-04-08 10:15:00 | 119.20 | 2025-04-09 09:15:00 | 117.84 | EXIT_EMA400 | -1.36 |
| SELL | 2025-08-21 12:15:00 | 136.81 | 2025-08-26 09:15:00 | 132.54 | TARGET | 4.27 |
| BUY | 2025-10-20 09:15:00 | 139.55 | 2025-10-30 12:15:00 | 148.62 | TARGET | 9.07 |
| BUY | 2025-12-30 10:15:00 | 150.14 | 2025-12-31 10:15:00 | 154.09 | TARGET | 3.95 |
| BUY | 2025-12-11 09:15:00 | 149.42 | 2026-01-02 11:15:00 | 156.26 | TARGET | 6.84 |
| BUY | 2025-12-09 11:15:00 | 149.97 | 2026-01-05 09:15:00 | 158.87 | TARGET | 8.90 |
| BUY | 2026-03-10 12:15:00 | 184.83 | 2026-03-13 09:15:00 | 176.81 | EXIT_EMA400 | -8.02 |
| BUY | 2026-03-12 10:15:00 | 179.20 | 2026-03-13 09:15:00 | 176.81 | EXIT_EMA400 | -2.39 |
