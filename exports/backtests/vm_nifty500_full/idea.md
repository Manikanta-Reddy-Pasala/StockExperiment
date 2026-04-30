# Vodafone Idea Ltd. (IDEA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4992 bars)
- **Last close:** 10.22
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -0.56
- **Avg P&L per closed trade:** -0.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 13:15:00 | 13.00 | 14.50 | 14.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 11:15:00 | 12.90 | 14.34 | 14.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 14.10 | 13.91 | 14.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 09:15:00 | 13.75 | 13.91 | 14.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-23 10:15:00 | 13.80 | 13.41 | 13.76 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 14.90 | 13.69 | 13.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 15.15 | 13.81 | 13.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 13.60 | 13.98 | 13.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 15.30 | 14.00 | 13.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 16.04 | 16.45 | 15.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-23 09:15:00 | 15.45 | 16.41 | 15.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 13.43 | 15.73 | 15.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 12.92 | 14.64 | 15.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 8.25 | 8.21 | 9.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 13:15:00 | 7.55 | 8.19 | 9.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 8.42 | 7.95 | 8.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-15 10:15:00 | 8.57 | 7.96 | 8.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 9.56 | 8.77 | 8.77 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 14:15:00 | 8.28 | 8.79 | 8.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 8.10 | 8.78 | 8.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 10:15:00 | 7.82 | 7.55 | 7.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 7.25 | 7.67 | 7.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 7.98 | 7.52 | 7.81 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 7.59 | 7.21 | 7.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 7.83 | 7.21 | 7.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 7.33 | 7.35 | 7.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 7.44 | 7.35 | 7.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 7.44 | 7.35 | 7.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 11:15:00 | 7.26 | 7.35 | 7.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 6.77 | 7.24 | 7.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 6.57 | 7.11 | 7.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 6.88 | 6.88 | 7.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 09:15:00 | 6.71 | 6.87 | 7.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-22 13:15:00 | 7.20 | 6.86 | 7.01 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 8.11 | 7.04 | 7.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 8.60 | 7.24 | 7.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 8.38 | 8.72 | 8.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 14:15:00 | 9.54 | 8.74 | 8.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 10.71 | 11.12 | 10.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-01 09:15:00 | 11.29 | 11.12 | 10.35 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 10.66 | 11.22 | 10.66 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 9.99 | 10.72 | 10.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 9.94 | 10.71 | 10.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 9.45 | 9.41 | 9.85 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-02 09:15:00 | 13.75 | 2024-04-12 09:15:00 | 12.53 | TARGET | 1.22 |
| BUY | 2024-06-06 09:15:00 | 15.30 | 2024-07-23 09:15:00 | 15.45 | EXIT_EMA400 | 0.15 |
| SELL | 2024-11-26 13:15:00 | 7.55 | 2025-01-15 10:15:00 | 8.57 | EXIT_EMA400 | -1.02 |
| SELL | 2025-04-07 09:15:00 | 7.25 | 2025-04-21 13:15:00 | 7.98 | EXIT_EMA400 | -0.73 |
| BUY | 2025-07-28 09:15:00 | 7.44 | 2025-07-28 11:15:00 | 7.26 | EXIT_EMA400 | -0.18 |
| SELL | 2025-08-21 09:15:00 | 6.71 | 2025-08-22 13:15:00 | 7.20 | EXIT_EMA400 | -0.49 |
| BUY | 2025-11-03 14:15:00 | 9.54 | 2026-01-19 09:15:00 | 10.66 | EXIT_EMA400 | 1.12 |
| BUY | 2026-01-01 09:15:00 | 11.29 | 2026-01-19 09:15:00 | 10.66 | EXIT_EMA400 | -0.63 |
