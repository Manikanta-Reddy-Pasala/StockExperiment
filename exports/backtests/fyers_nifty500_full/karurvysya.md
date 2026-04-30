# Karur Vysya Bank Ltd. (KARURVYSYA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 292.95
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
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 3.56
- **Avg P&L per closed trade:** 0.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 13:15:00 | 165.00 | 177.76 | 177.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 163.42 | 177.12 | 177.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 179.83 | 174.19 | 175.77 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 186.29 | 177.01 | 177.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 189.48 | 177.43 | 177.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 178.76 | 182.73 | 180.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-25 09:15:00 | 185.43 | 181.32 | 180.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-20 13:15:00 | 187.62 | 192.30 | 188.03 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 176.54 | 185.48 | 185.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 174.61 | 185.20 | 185.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 14:15:00 | 183.53 | 183.38 | 184.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-17 10:15:00 | 182.10 | 183.38 | 184.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 182.10 | 183.38 | 184.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-17 12:15:00 | 180.90 | 183.34 | 184.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-17 13:15:00 | 184.58 | 183.35 | 184.33 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 189.71 | 185.06 | 185.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 191.50 | 185.35 | 185.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 11:15:00 | 189.64 | 189.66 | 187.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-11 15:15:00 | 191.25 | 189.64 | 187.80 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-12 09:15:00 | 185.76 | 189.60 | 187.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 173.66 | 186.55 | 186.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 169.75 | 184.36 | 185.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 173.24 | 172.73 | 177.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 167.56 | 174.66 | 177.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 175.98 | 174.37 | 177.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 177.47 | 174.42 | 177.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 181.68 | 178.65 | 178.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 183.36 | 178.70 | 178.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 178.93 | 179.00 | 178.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 14:15:00 | 180.04 | 179.01 | 178.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 180.04 | 179.01 | 178.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-02 15:15:00 | 178.68 | 179.00 | 178.82 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 175.00 | 178.66 | 178.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 173.52 | 178.37 | 178.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 178.70 | 177.94 | 178.28 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 183.18 | 178.61 | 178.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 185.48 | 178.92 | 178.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 12:15:00 | 183.10 | 183.27 | 181.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 14:15:00 | 184.78 | 183.29 | 181.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 213.20 | 217.29 | 213.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-28 12:15:00 | 212.50 | 217.24 | 213.10 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 14:15:00 | 281.45 | 287.62 | 287.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 13:15:00 | 277.25 | 287.26 | 287.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 284.55 | 284.43 | 285.86 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 299.20 | 287.13 | 287.11 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-25 09:15:00 | 185.43 | 2024-11-29 09:15:00 | 201.71 | TARGET | 16.28 |
| SELL | 2025-01-17 10:15:00 | 182.10 | 2025-01-17 13:15:00 | 184.58 | EXIT_EMA400 | -2.48 |
| SELL | 2025-01-17 12:15:00 | 180.90 | 2025-01-17 13:15:00 | 184.58 | EXIT_EMA400 | -3.68 |
| BUY | 2025-02-11 15:15:00 | 191.25 | 2025-02-12 09:15:00 | 185.76 | EXIT_EMA400 | -5.49 |
| SELL | 2025-04-07 09:15:00 | 167.56 | 2025-04-08 12:15:00 | 177.47 | EXIT_EMA400 | -9.91 |
| BUY | 2025-05-02 14:15:00 | 180.04 | 2025-05-02 15:15:00 | 178.68 | EXIT_EMA400 | -1.36 |
| BUY | 2025-05-29 14:15:00 | 184.78 | 2025-06-02 10:15:00 | 194.98 | TARGET | 10.20 |
