# Devyani International Ltd. (DEVYANI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 125.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 2.04
- **Avg P&L per closed trade:** 0.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 162.67 | 180.73 | 180.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 161.39 | 180.54 | 180.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 176.00 | 174.68 | 177.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 09:15:00 | 172.68 | 174.66 | 177.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 11:15:00 | 173.85 | 168.22 | 171.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 182.63 | 173.03 | 173.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 186.41 | 174.69 | 173.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 180.30 | 180.57 | 177.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-13 09:15:00 | 183.79 | 180.60 | 177.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-13 13:15:00 | 177.17 | 180.55 | 177.45 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 09:15:00 | 169.58 | 176.64 | 176.65 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 192.61 | 176.81 | 176.74 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 166.46 | 176.90 | 176.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 12:15:00 | 164.24 | 176.55 | 176.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 174.09 | 172.54 | 174.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-28 09:15:00 | 168.55 | 173.44 | 174.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 174.64 | 171.75 | 173.63 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 177.04 | 165.09 | 165.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 177.99 | 165.68 | 165.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 168.24 | 168.54 | 166.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 10:15:00 | 169.96 | 168.55 | 166.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 172.00 | 174.73 | 171.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-28 12:15:00 | 173.57 | 174.62 | 171.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-29 13:15:00 | 171.33 | 174.50 | 171.38 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 166.28 | 170.15 | 170.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 165.82 | 170.07 | 170.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 173.73 | 170.14 | 170.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 175.65 | 170.46 | 170.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 12:15:00 | 172.25 | 172.37 | 171.44 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 162.55 | 170.69 | 170.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 162.25 | 170.60 | 170.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 175.42 | 168.54 | 168.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 180.05 | 168.91 | 168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 175.88 | 176.14 | 173.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-19 14:15:00 | 177.70 | 176.12 | 173.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 173.22 | 176.14 | 173.57 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 165.00 | 171.74 | 171.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 164.24 | 171.41 | 171.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 139.12 | 137.97 | 146.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 13:15:00 | 133.96 | 140.21 | 144.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 10:15:00 | 134.48 | 126.51 | 134.26 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 09:15:00 | 172.68 | 2024-11-21 09:15:00 | 159.73 | TARGET | 12.95 |
| BUY | 2025-01-13 09:15:00 | 183.79 | 2025-01-13 13:15:00 | 177.17 | EXIT_EMA400 | -6.62 |
| SELL | 2025-02-28 09:15:00 | 168.55 | 2025-03-06 09:15:00 | 174.64 | EXIT_EMA400 | -6.09 |
| BUY | 2025-05-09 10:15:00 | 169.96 | 2025-05-09 15:15:00 | 179.00 | TARGET | 9.04 |
| BUY | 2025-05-28 12:15:00 | 173.57 | 2025-05-29 13:15:00 | 171.33 | EXIT_EMA400 | -2.24 |
| BUY | 2025-09-19 14:15:00 | 177.70 | 2025-09-25 14:15:00 | 173.22 | EXIT_EMA400 | -4.48 |
| SELL | 2026-01-09 13:15:00 | 133.96 | 2026-02-05 10:15:00 | 134.48 | EXIT_EMA400 | -0.52 |
