# Vedanta Ltd. (VEDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 273.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 22.33
- **Avg P&L per closed trade:** 3.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 160.79 | 165.09 | 165.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 158.80 | 164.63 | 164.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 11:15:00 | 163.73 | 163.69 | 164.35 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 171.03 | 164.95 | 164.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 171.82 | 165.41 | 165.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 165.07 | 169.47 | 167.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-10 10:15:00 | 165.64 | 169.43 | 167.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 165.64 | 169.43 | 167.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-10 11:15:00 | 165.49 | 169.39 | 167.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 12:15:00 | 163.95 | 174.24 | 174.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 13:15:00 | 162.53 | 174.13 | 174.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 171.91 | 171.06 | 172.38 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 186.16 | 173.43 | 173.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 187.27 | 173.69 | 173.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 178.56 | 180.60 | 177.62 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 172.04 | 175.57 | 175.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 171.61 | 175.53 | 175.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 170.49 | 170.17 | 172.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 167.79 | 170.35 | 172.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-07 10:15:00 | 170.21 | 166.41 | 169.17 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 174.03 | 165.72 | 165.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 176.12 | 166.18 | 165.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 165.36 | 168.18 | 167.09 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 137.66 | 165.85 | 165.98 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 168.61 | 161.11 | 161.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 169.98 | 162.02 | 161.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 09:15:00 | 162.60 | 162.29 | 161.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 162.57 | 162.30 | 161.78 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-04 09:15:00 | 162.62 | 162.31 | 161.79 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 165.32 | 166.99 | 164.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 14:15:00 | 164.70 | 166.97 | 164.72 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 161.46 | 166.20 | 166.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 161.24 | 165.51 | 165.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 161.44 | 164.97 | 165.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-19 09:15:00 | 167.81 | 164.70 | 165.35 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 173.07 | 165.27 | 165.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 175.49 | 165.46 | 165.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 167.08 | 167.46 | 166.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 169.94 | 167.49 | 166.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 255.36 | 259.89 | 249.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 13:15:00 | 249.38 | 259.62 | 249.51 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-10 10:15:00 | 165.64 | 2024-09-10 11:15:00 | 165.49 | EXIT_EMA400 | -0.15 |
| SELL | 2025-01-22 09:15:00 | 167.79 | 2025-02-03 09:15:00 | 154.17 | TARGET | 13.62 |
| BUY | 2025-06-03 09:15:00 | 162.60 | 2025-06-05 14:15:00 | 165.11 | TARGET | 2.51 |
| BUY | 2025-06-04 09:15:00 | 162.62 | 2025-06-05 14:15:00 | 165.10 | TARGET | 2.48 |
| SELL | 2025-08-14 09:15:00 | 161.44 | 2025-08-19 09:15:00 | 167.81 | EXIT_EMA400 | -6.37 |
| BUY | 2025-09-29 09:15:00 | 169.94 | 2025-10-09 10:15:00 | 180.18 | TARGET | 10.24 |
