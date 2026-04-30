# Karur Vysya Bank Ltd. (KARURVYSYA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 293.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 3.61
- **Avg P&L per closed trade:** 0.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 165.62 | 177.64 | 177.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 163.75 | 177.26 | 177.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 179.92 | 174.18 | 175.75 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 186.35 | 177.01 | 176.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 189.47 | 177.43 | 177.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 178.84 | 182.60 | 180.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-25 09:15:00 | 185.43 | 181.23 | 179.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-20 13:15:00 | 187.71 | 192.27 | 187.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 176.67 | 185.47 | 185.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 174.71 | 185.19 | 185.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 14:15:00 | 183.44 | 183.39 | 184.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-17 10:15:00 | 182.10 | 183.38 | 184.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 182.10 | 183.38 | 184.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-17 12:15:00 | 180.90 | 183.34 | 184.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-17 13:15:00 | 184.58 | 183.36 | 184.31 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 189.58 | 185.07 | 185.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 191.50 | 185.35 | 185.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 188.00 | 189.21 | 187.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-11 15:15:00 | 191.29 | 189.21 | 187.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-12 09:15:00 | 185.76 | 189.18 | 187.49 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 173.59 | 186.31 | 186.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 169.75 | 184.16 | 185.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 173.25 | 172.66 | 177.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 167.30 | 174.62 | 177.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 175.98 | 174.33 | 177.06 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 177.47 | 174.38 | 177.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 181.36 | 178.55 | 178.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 183.36 | 178.68 | 178.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 178.94 | 178.97 | 178.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 14:15:00 | 180.04 | 178.98 | 178.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 180.04 | 178.98 | 178.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-02 15:15:00 | 178.68 | 178.98 | 178.77 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 176.69 | 178.57 | 178.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 12:15:00 | 176.20 | 178.54 | 178.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 178.70 | 177.93 | 178.23 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 184.92 | 178.51 | 178.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 185.48 | 178.91 | 178.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 12:15:00 | 183.10 | 183.27 | 181.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 14:15:00 | 184.79 | 183.29 | 181.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 213.20 | 217.31 | 213.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-28 12:15:00 | 212.50 | 217.26 | 213.11 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 12:15:00 | 280.10 | 287.29 | 287.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 13:15:00 | 277.25 | 287.19 | 287.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 284.70 | 284.69 | 285.87 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 295.30 | 286.96 | 286.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 299.20 | 287.32 | 287.12 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-25 09:15:00 | 185.43 | 2024-11-29 09:15:00 | 201.95 | TARGET | 16.52 |
| SELL | 2025-01-17 10:15:00 | 182.10 | 2025-01-17 13:15:00 | 184.58 | EXIT_EMA400 | -2.48 |
| SELL | 2025-01-17 12:15:00 | 180.90 | 2025-01-17 13:15:00 | 184.58 | EXIT_EMA400 | -3.68 |
| BUY | 2025-02-11 15:15:00 | 191.29 | 2025-02-12 09:15:00 | 185.76 | EXIT_EMA400 | -5.53 |
| SELL | 2025-04-07 09:15:00 | 167.30 | 2025-04-08 12:15:00 | 177.47 | EXIT_EMA400 | -10.17 |
| BUY | 2025-05-02 14:15:00 | 180.04 | 2025-05-02 15:15:00 | 178.68 | EXIT_EMA400 | -1.36 |
| BUY | 2025-05-29 14:15:00 | 184.79 | 2025-06-02 10:15:00 | 195.11 | TARGET | 10.32 |
