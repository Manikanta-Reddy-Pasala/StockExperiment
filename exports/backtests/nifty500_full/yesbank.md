# Yes Bank Ltd. (YESBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 19.93
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -1.05
- **Avg P&L per closed trade:** -0.21

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 13:15:00 | 16.05 | 17.21 | 17.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 15.80 | 17.18 | 17.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 16.80 | 16.76 | 16.96 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 19.35 | 17.12 | 17.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 19.75 | 17.17 | 17.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 09:15:00 | 23.45 | 23.62 | 22.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-06 09:15:00 | 25.05 | 23.61 | 22.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-28 10:15:00 | 24.45 | 26.06 | 24.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 15:15:00 | 22.30 | 24.28 | 24.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 22.15 | 23.42 | 23.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 15:15:00 | 23.30 | 23.26 | 23.62 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 26.60 | 23.79 | 23.79 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 23.75 | 24.43 | 24.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 23.59 | 24.42 | 24.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 10:15:00 | 23.91 | 23.88 | 24.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 14:15:00 | 23.46 | 23.87 | 24.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 11:15:00 | 21.24 | 20.36 | 21.11 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 20.45 | 17.95 | 17.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 21.63 | 18.55 | 18.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 20.44 | 20.51 | 19.73 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 18.61 | 19.82 | 19.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 18.53 | 19.79 | 19.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 19.36 | 19.29 | 19.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 10:15:00 | 19.13 | 19.33 | 19.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-29 11:15:00 | 19.46 | 19.28 | 19.45 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 20.39 | 19.58 | 19.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 20.75 | 19.61 | 19.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 22.42 | 22.47 | 21.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-17 09:15:00 | 22.96 | 22.47 | 21.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 11:15:00 | 22.16 | 22.61 | 22.20 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 21.39 | 21.98 | 21.98 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 22.83 | 21.98 | 21.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 23.15 | 21.99 | 21.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 22.19 | 22.45 | 22.25 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 21.16 | 22.09 | 22.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 20.91 | 21.98 | 22.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 14:15:00 | 19.02 | 19.01 | 19.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 18.74 | 19.01 | 19.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 19.73 | 19.03 | 19.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 19.98 | 19.04 | 19.79 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-06 09:15:00 | 25.05 | 2024-02-28 10:15:00 | 24.45 | EXIT_EMA400 | -0.60 |
| SELL | 2024-09-16 14:15:00 | 23.46 | 2024-10-04 09:15:00 | 21.54 | TARGET | 1.92 |
| SELL | 2025-08-26 10:15:00 | 19.13 | 2025-08-29 11:15:00 | 19.46 | EXIT_EMA400 | -0.33 |
| BUY | 2025-11-17 09:15:00 | 22.96 | 2025-12-08 11:15:00 | 22.16 | EXIT_EMA400 | -0.80 |
| SELL | 2026-04-13 09:15:00 | 18.74 | 2026-04-16 10:15:00 | 19.98 | EXIT_EMA400 | -1.24 |
