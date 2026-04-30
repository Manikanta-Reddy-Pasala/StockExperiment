# Godawari Power & Ispat Ltd. (GPIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 296.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 46.52
- **Avg P&L per closed trade:** 4.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 134.52 | 146.27 | 146.28 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 11:15:00 | 152.26 | 145.93 | 145.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 159.49 | 146.89 | 146.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 175.39 | 177.00 | 168.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-29 09:15:00 | 182.56 | 177.03 | 168.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-13 09:15:00 | 208.53 | 218.70 | 210.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 184.70 | 204.99 | 205.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 184.20 | 204.79 | 204.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 194.29 | 192.17 | 196.72 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 09:15:00 | 212.00 | 200.15 | 200.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 190.85 | 200.13 | 200.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 186.95 | 197.38 | 198.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 191.00 | 188.69 | 193.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 14:15:00 | 181.42 | 190.97 | 193.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 191.18 | 189.65 | 192.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 09:15:00 | 187.27 | 189.59 | 192.00 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-02 14:15:00 | 192.54 | 189.55 | 191.92 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 220.28 | 193.89 | 193.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 230.10 | 195.85 | 194.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 13:15:00 | 211.05 | 211.60 | 204.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-01 14:15:00 | 213.25 | 209.90 | 204.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 204.00 | 209.76 | 205.26 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 12:15:00 | 186.54 | 202.26 | 202.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 14:15:00 | 185.94 | 201.95 | 202.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 188.05 | 187.69 | 193.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 183.72 | 187.63 | 192.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 181.60 | 173.06 | 181.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 193.35 | 183.24 | 183.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 194.50 | 183.36 | 183.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 188.83 | 189.18 | 186.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 190.10 | 185.97 | 185.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-30 14:15:00 | 189.42 | 193.94 | 190.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 178.65 | 189.17 | 189.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 10:15:00 | 176.32 | 188.24 | 188.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 13:15:00 | 185.41 | 187.16 | 188.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 186.20 | 185.36 | 186.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-11 11:15:00 | 187.02 | 185.38 | 186.79 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 192.23 | 187.70 | 187.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 194.40 | 188.03 | 187.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 189.34 | 189.41 | 188.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 09:15:00 | 190.99 | 189.42 | 188.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 190.99 | 189.42 | 188.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 12:15:00 | 194.81 | 189.53 | 188.72 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 190.43 | 190.37 | 189.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-06 10:15:00 | 188.57 | 190.36 | 189.19 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 230.01 | 246.43 | 246.50 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 256.34 | 245.74 | 245.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 261.00 | 246.00 | 245.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 256.35 | 257.01 | 252.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 261.75 | 257.01 | 252.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 254.45 | 257.66 | 253.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 10:15:00 | 251.20 | 257.59 | 253.20 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-29 09:15:00 | 182.56 | 2024-06-18 09:15:00 | 224.85 | TARGET | 42.29 |
| SELL | 2024-11-21 14:15:00 | 181.42 | 2024-12-02 14:15:00 | 192.54 | EXIT_EMA400 | -11.12 |
| SELL | 2024-12-02 09:15:00 | 187.27 | 2024-12-02 14:15:00 | 192.54 | EXIT_EMA400 | -5.27 |
| BUY | 2025-01-01 14:15:00 | 213.25 | 2025-01-06 09:15:00 | 204.00 | EXIT_EMA400 | -9.25 |
| SELL | 2025-02-10 09:15:00 | 183.72 | 2025-02-28 12:15:00 | 156.30 | TARGET | 27.42 |
| BUY | 2025-05-12 14:15:00 | 190.10 | 2025-05-14 11:15:00 | 203.93 | TARGET | 13.83 |
| SELL | 2025-07-01 13:15:00 | 185.41 | 2025-07-11 11:15:00 | 187.02 | EXIT_EMA400 | -1.61 |
| BUY | 2025-08-04 09:15:00 | 190.99 | 2025-08-05 09:15:00 | 198.00 | TARGET | 7.01 |
| BUY | 2025-08-04 12:15:00 | 194.81 | 2025-08-06 10:15:00 | 188.57 | EXIT_EMA400 | -6.24 |
| BUY | 2026-01-14 09:15:00 | 261.75 | 2026-01-19 10:15:00 | 251.20 | EXIT_EMA400 | -10.55 |
