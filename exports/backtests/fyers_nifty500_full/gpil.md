# Godawari Power & Ispat Ltd. (GPIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 296.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 12.10
- **Avg P&L per closed trade:** 1.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 190.20 | 207.31 | 207.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 189.14 | 207.13 | 207.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 194.29 | 192.20 | 197.39 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 14:15:00 | 206.85 | 201.21 | 201.19 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 194.50 | 201.10 | 201.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 191.95 | 201.01 | 201.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 190.10 | 188.67 | 193.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 14:15:00 | 181.39 | 191.04 | 193.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 14:15:00 | 192.54 | 189.58 | 192.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 221.26 | 194.18 | 194.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 229.83 | 195.87 | 195.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 13:15:00 | 211.05 | 211.60 | 204.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-01 14:15:00 | 213.25 | 209.90 | 205.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 204.00 | 209.76 | 205.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 12:15:00 | 186.54 | 202.28 | 202.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 14:15:00 | 186.05 | 201.97 | 202.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 187.65 | 187.48 | 192.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 09:15:00 | 184.36 | 187.45 | 192.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 181.60 | 173.00 | 181.28 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 192.58 | 183.14 | 183.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 194.50 | 183.35 | 183.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 188.83 | 189.18 | 186.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 190.10 | 185.97 | 185.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-30 14:15:00 | 189.42 | 193.93 | 190.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 178.65 | 189.19 | 189.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 176.20 | 189.06 | 189.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 13:15:00 | 185.41 | 187.16 | 187.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 186.20 | 185.36 | 186.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-11 11:15:00 | 187.02 | 185.38 | 186.79 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 192.23 | 187.68 | 187.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 194.40 | 188.02 | 187.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 188.10 | 188.14 | 187.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 11:15:00 | 189.76 | 188.20 | 187.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 15:15:00 | 188.27 | 189.38 | 188.62 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 230.20 | 246.41 | 246.49 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 256.34 | 245.73 | 245.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 261.00 | 245.99 | 245.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 256.35 | 257.03 | 252.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 261.75 | 257.02 | 252.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 254.60 | 257.67 | 253.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 10:15:00 | 251.20 | 257.60 | 253.21 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-21 14:15:00 | 181.39 | 2024-12-02 14:15:00 | 192.54 | EXIT_EMA400 | -11.15 |
| BUY | 2025-01-01 14:15:00 | 213.25 | 2025-01-06 09:15:00 | 204.00 | EXIT_EMA400 | -9.25 |
| SELL | 2025-02-07 09:15:00 | 184.36 | 2025-02-28 09:15:00 | 159.04 | TARGET | 25.32 |
| BUY | 2025-05-12 14:15:00 | 190.10 | 2025-05-14 11:15:00 | 204.03 | TARGET | 13.93 |
| SELL | 2025-07-01 13:15:00 | 185.41 | 2025-07-11 11:15:00 | 187.02 | EXIT_EMA400 | -1.61 |
| BUY | 2025-07-29 11:15:00 | 189.76 | 2025-07-30 09:15:00 | 195.17 | TARGET | 5.41 |
| BUY | 2026-01-14 09:15:00 | 261.75 | 2026-01-19 10:15:00 | 251.20 | EXIT_EMA400 | -10.55 |
