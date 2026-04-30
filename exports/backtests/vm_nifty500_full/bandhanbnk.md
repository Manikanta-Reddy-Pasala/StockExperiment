# Bandhan Bank Ltd. (BANDHANBNK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 199.72
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 5 |
| EXIT | 6 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 9 / 2
- **Target hits / EMA400 exits:** 9 / 2
- **Total realized P&L (per unit):** 106.41
- **Avg P&L per closed trade:** 9.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 10:15:00 | 251.30 | 236.58 | 236.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 252.35 | 238.39 | 237.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 12:15:00 | 245.10 | 245.98 | 242.53 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 15:15:00 | 218.25 | 239.98 | 240.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 213.40 | 239.71 | 239.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 12:15:00 | 222.55 | 221.15 | 227.09 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 12:15:00 | 252.50 | 230.49 | 230.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 11:15:00 | 254.80 | 231.85 | 231.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 13:15:00 | 233.75 | 234.64 | 232.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 14:15:00 | 238.95 | 234.69 | 232.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 236.80 | 240.24 | 236.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-08 12:15:00 | 236.05 | 240.20 | 236.43 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 14:15:00 | 224.45 | 234.14 | 234.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 221.20 | 233.72 | 233.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 10:15:00 | 231.95 | 231.34 | 232.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-05 09:15:00 | 223.10 | 231.17 | 232.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 200.00 | 192.18 | 201.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-05 11:15:00 | 198.25 | 192.31 | 201.86 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-28 09:15:00 | 190.90 | 185.66 | 189.99 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 206.21 | 191.70 | 191.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 12:15:00 | 206.80 | 191.85 | 191.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 195.61 | 199.94 | 196.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 09:15:00 | 211.86 | 196.21 | 195.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-08 14:15:00 | 198.95 | 202.04 | 199.35 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 184.49 | 200.10 | 200.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 182.06 | 196.30 | 197.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 177.08 | 175.98 | 182.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 09:15:00 | 174.67 | 176.15 | 182.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-07 10:15:00 | 150.82 | 144.32 | 150.72 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 169.33 | 150.26 | 150.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 170.61 | 150.66 | 150.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 157.31 | 157.71 | 154.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 158.60 | 157.69 | 154.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 157.00 | 157.88 | 154.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 11:15:00 | 157.27 | 157.85 | 154.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 175.12 | 179.66 | 175.44 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 166.72 | 172.71 | 172.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 166.19 | 172.59 | 172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 172.91 | 171.34 | 171.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 168.23 | 172.03 | 172.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 167.79 | 166.70 | 168.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-18 11:15:00 | 166.91 | 166.70 | 168.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 165.18 | 164.14 | 166.59 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-06 12:15:00 | 164.61 | 164.17 | 166.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 165.36 | 164.20 | 166.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 10:15:00 | 165.01 | 164.21 | 166.53 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-10 12:15:00 | 166.58 | 163.98 | 166.16 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 166.70 | 151.12 | 151.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 168.54 | 151.74 | 151.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 169.12 | 171.77 | 164.88 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 142.15 | 160.75 | 160.78 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 170.71 | 160.63 | 160.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 171.55 | 160.74 | 160.66 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 14:15:00 | 238.95 | 2024-01-03 12:15:00 | 257.54 | TARGET | 18.59 |
| SELL | 2024-02-05 09:15:00 | 223.10 | 2024-02-13 09:15:00 | 194.94 | TARGET | 28.16 |
| SELL | 2024-04-05 11:15:00 | 198.25 | 2024-04-08 09:15:00 | 187.43 | TARGET | 10.82 |
| BUY | 2024-07-29 09:15:00 | 211.86 | 2024-08-08 14:15:00 | 198.95 | EXIT_EMA400 | -12.91 |
| SELL | 2024-12-06 09:15:00 | 174.67 | 2025-01-06 14:15:00 | 151.81 | TARGET | 22.86 |
| BUY | 2025-05-09 11:15:00 | 157.27 | 2025-05-12 09:15:00 | 164.16 | TARGET | 6.89 |
| BUY | 2025-05-07 10:15:00 | 158.60 | 2025-05-16 09:15:00 | 170.38 | TARGET | 11.78 |
| SELL | 2025-09-18 11:15:00 | 166.91 | 2025-09-23 09:15:00 | 161.43 | TARGET | 5.48 |
| SELL | 2025-08-26 09:15:00 | 168.23 | 2025-09-26 09:15:00 | 156.10 | TARGET | 12.13 |
| SELL | 2025-10-07 10:15:00 | 165.01 | 2025-10-08 14:15:00 | 160.44 | TARGET | 4.57 |
| SELL | 2025-10-06 12:15:00 | 164.61 | 2025-10-10 12:15:00 | 166.58 | EXIT_EMA400 | -1.97 |
