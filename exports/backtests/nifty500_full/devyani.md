# Devyani International Ltd. (DEVYANI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 124.73
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 1 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -24.75
- **Avg P&L per closed trade:** -2.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 180.55 | 202.48 | 202.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 179.05 | 202.25 | 202.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 11:15:00 | 185.95 | 185.10 | 190.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-13 10:15:00 | 184.35 | 185.36 | 189.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-19 09:15:00 | 193.10 | 185.05 | 189.17 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 180.10 | 160.94 | 160.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 180.94 | 168.78 | 166.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 172.00 | 172.51 | 169.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 14:15:00 | 180.29 | 172.56 | 169.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-12 09:15:00 | 168.91 | 173.30 | 170.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 162.79 | 180.72 | 180.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 161.40 | 180.53 | 180.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 176.00 | 174.75 | 177.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 09:15:00 | 172.68 | 174.73 | 177.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 11:15:00 | 173.85 | 168.25 | 171.83 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 183.02 | 173.14 | 173.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 186.41 | 174.69 | 173.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 180.30 | 180.57 | 177.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-13 09:15:00 | 183.79 | 180.60 | 177.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-13 13:15:00 | 177.17 | 180.55 | 177.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 166.46 | 176.97 | 176.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 10:15:00 | 165.22 | 176.86 | 176.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 174.09 | 172.59 | 174.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-28 09:15:00 | 168.55 | 173.49 | 174.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 174.64 | 171.78 | 173.67 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 176.97 | 165.10 | 165.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 177.99 | 165.68 | 165.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 168.24 | 168.53 | 166.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 10:15:00 | 170.00 | 168.55 | 166.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-29 13:15:00 | 171.33 | 174.51 | 171.39 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 166.28 | 170.15 | 170.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 165.82 | 170.07 | 170.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 172.89 | 169.84 | 169.99 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 173.73 | 170.15 | 170.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 174.80 | 170.41 | 170.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 12:15:00 | 172.25 | 172.36 | 171.44 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 162.55 | 170.69 | 170.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 162.25 | 170.61 | 170.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 165.74 | 163.57 | 166.54 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 175.42 | 168.55 | 168.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 180.10 | 168.92 | 168.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 175.88 | 176.15 | 173.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-19 14:15:00 | 177.70 | 176.13 | 173.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 173.22 | 176.14 | 173.57 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 14:15:00 | 165.03 | 171.81 | 171.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 164.24 | 171.41 | 171.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 139.12 | 137.97 | 146.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 13:15:00 | 133.96 | 140.22 | 144.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 134.40 | 127.14 | 134.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-06 09:15:00 | 127.00 | 127.40 | 134.76 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-10 10:15:00 | 135.40 | 127.95 | 134.51 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-13 10:15:00 | 184.35 | 2023-12-19 09:15:00 | 193.10 | EXIT_EMA400 | -8.75 |
| BUY | 2024-08-05 14:15:00 | 180.29 | 2024-08-12 09:15:00 | 168.91 | EXIT_EMA400 | -11.38 |
| SELL | 2024-11-12 09:15:00 | 172.68 | 2024-11-21 09:15:00 | 159.41 | TARGET | 13.27 |
| BUY | 2025-01-13 09:15:00 | 183.79 | 2025-01-13 13:15:00 | 177.17 | EXIT_EMA400 | -6.62 |
| SELL | 2025-02-28 09:15:00 | 168.55 | 2025-03-06 09:15:00 | 174.64 | EXIT_EMA400 | -6.09 |
| BUY | 2025-05-09 10:15:00 | 170.00 | 2025-05-09 15:15:00 | 179.13 | TARGET | 9.13 |
| BUY | 2025-09-19 14:15:00 | 177.70 | 2025-09-25 14:15:00 | 173.22 | EXIT_EMA400 | -4.48 |
| SELL | 2026-01-09 13:15:00 | 133.96 | 2026-02-10 10:15:00 | 135.40 | EXIT_EMA400 | -1.44 |
| SELL | 2026-02-06 09:15:00 | 127.00 | 2026-02-10 10:15:00 | 135.40 | EXIT_EMA400 | -8.40 |
