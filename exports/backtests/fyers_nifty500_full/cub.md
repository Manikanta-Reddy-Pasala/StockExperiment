# City Union Bank Ltd. (CUB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 270.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 25.08
- **Avg P&L per closed trade:** 3.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 154.50 | 163.97 | 164.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 154.22 | 163.78 | 163.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 168.11 | 159.79 | 161.62 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 174.94 | 163.19 | 163.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 176.15 | 163.32 | 163.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 12:15:00 | 169.74 | 170.23 | 167.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-13 13:15:00 | 171.77 | 170.25 | 167.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 168.06 | 170.53 | 167.83 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-11-21 12:15:00 | 170.63 | 170.52 | 167.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 174.51 | 179.37 | 175.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 165.88 | 173.23 | 173.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 163.72 | 171.84 | 172.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 171.60 | 171.28 | 172.15 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 173.63 | 172.76 | 172.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 12:15:00 | 174.05 | 172.77 | 172.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 170.31 | 172.73 | 172.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 168.27 | 172.67 | 172.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 15:15:00 | 156.50 | 156.33 | 161.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 151.84 | 158.20 | 160.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 158.96 | 158.05 | 160.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-08 10:15:00 | 157.52 | 158.04 | 160.72 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 160.00 | 158.09 | 160.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-08 13:15:00 | 159.74 | 158.10 | 160.71 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 160.40 | 158.12 | 160.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 15:15:00 | 161.55 | 158.16 | 160.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 182.40 | 162.75 | 162.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 184.25 | 162.97 | 162.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 192.86 | 193.07 | 184.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 12:15:00 | 195.25 | 193.10 | 184.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 206.15 | 208.99 | 201.94 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 11:15:00 | 210.24 | 208.91 | 202.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 206.99 | 211.76 | 206.89 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-25 12:15:00 | 204.57 | 211.61 | 206.89 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 246.70 | 275.28 | 275.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 246.30 | 275.00 | 275.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 257.50 | 252.87 | 260.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 11:15:00 | 253.02 | 253.09 | 260.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-17 15:15:00 | 259.51 | 252.97 | 259.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-21 12:15:00 | 170.63 | 2024-11-26 14:15:00 | 178.93 | TARGET | 8.30 |
| BUY | 2024-11-13 13:15:00 | 171.77 | 2024-12-04 09:15:00 | 185.03 | TARGET | 13.26 |
| SELL | 2025-04-07 09:15:00 | 151.84 | 2025-04-08 15:15:00 | 161.55 | EXIT_EMA400 | -9.71 |
| SELL | 2025-04-08 10:15:00 | 157.52 | 2025-04-08 15:15:00 | 161.55 | EXIT_EMA400 | -4.03 |
| SELL | 2025-04-08 13:15:00 | 159.74 | 2025-04-08 15:15:00 | 161.55 | EXIT_EMA400 | -1.81 |
| BUY | 2025-06-13 12:15:00 | 195.25 | 2025-07-01 13:15:00 | 226.48 | TARGET | 31.23 |
| BUY | 2025-07-31 11:15:00 | 210.24 | 2025-08-25 12:15:00 | 204.57 | EXIT_EMA400 | -5.67 |
| SELL | 2026-04-09 11:15:00 | 253.02 | 2026-04-17 15:15:00 | 259.51 | EXIT_EMA400 | -6.49 |
