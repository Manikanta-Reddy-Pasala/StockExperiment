# JM Financial Ltd. (JMFINANCIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 139.07
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 31.26
- **Avg P&L per closed trade:** 3.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 12:15:00 | 79.65 | 98.69 | 98.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 78.70 | 97.93 | 98.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 10:15:00 | 83.30 | 83.23 | 88.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-12 12:15:00 | 82.90 | 83.23 | 87.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-30 10:15:00 | 88.80 | 82.47 | 86.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 15:15:00 | 91.00 | 83.45 | 83.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 91.57 | 84.01 | 83.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 95.73 | 96.62 | 92.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-26 11:15:00 | 99.30 | 95.06 | 92.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-13 09:15:00 | 132.31 | 140.82 | 133.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 122.95 | 133.33 | 133.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 122.20 | 133.22 | 133.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 117.09 | 115.92 | 121.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 111.85 | 115.92 | 121.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 100.33 | 95.81 | 101.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-16 12:15:00 | 99.71 | 95.85 | 101.12 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-21 09:15:00 | 102.39 | 96.31 | 101.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 115.59 | 102.93 | 102.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 118.04 | 103.32 | 103.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 161.15 | 162.16 | 148.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 176.94 | 159.60 | 151.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 173.19 | 178.52 | 170.49 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-23 13:15:00 | 173.50 | 177.87 | 170.58 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 171.08 | 177.64 | 170.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-24 12:15:00 | 172.16 | 177.52 | 170.62 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 171.24 | 177.34 | 170.64 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-25 10:15:00 | 169.41 | 177.20 | 170.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 150.13 | 169.54 | 169.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 148.66 | 169.15 | 169.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 157.00 | 154.24 | 160.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 146.11 | 153.81 | 159.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 137.82 | 133.17 | 140.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 10:15:00 | 140.60 | 133.24 | 140.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 135.68 | 131.45 | 131.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 136.65 | 131.50 | 131.46 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-12 12:15:00 | 82.90 | 2024-04-30 10:15:00 | 88.80 | EXIT_EMA400 | -5.90 |
| BUY | 2024-08-26 11:15:00 | 99.30 | 2024-09-03 14:15:00 | 118.76 | TARGET | 19.46 |
| SELL | 2025-02-10 09:15:00 | 111.85 | 2025-04-07 09:15:00 | 82.61 | TARGET | 29.24 |
| SELL | 2025-04-16 12:15:00 | 99.71 | 2025-04-21 09:15:00 | 102.39 | EXIT_EMA400 | -2.68 |
| BUY | 2025-08-13 09:15:00 | 176.94 | 2025-09-25 10:15:00 | 169.41 | EXIT_EMA400 | -7.53 |
| BUY | 2025-09-23 13:15:00 | 173.50 | 2025-09-25 10:15:00 | 169.41 | EXIT_EMA400 | -4.09 |
| BUY | 2025-09-24 12:15:00 | 172.16 | 2025-09-25 10:15:00 | 169.41 | EXIT_EMA400 | -2.75 |
| SELL | 2025-12-04 09:15:00 | 146.11 | 2026-02-10 10:15:00 | 140.60 | EXIT_EMA400 | 5.51 |
