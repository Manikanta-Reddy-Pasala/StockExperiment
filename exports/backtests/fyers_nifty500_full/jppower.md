# Jaiprakash Power Ventures Ltd. (JPPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 19.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -1.62
- **Avg P&L per closed trade:** -0.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 18.01 | 18.78 | 18.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 17.84 | 18.77 | 18.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 11:15:00 | 18.61 | 18.48 | 18.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 10:15:00 | 18.14 | 18.48 | 18.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 17.94 | 17.74 | 18.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-24 12:15:00 | 17.72 | 17.74 | 18.07 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 17.45 | 17.71 | 18.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-30 11:15:00 | 18.26 | 17.71 | 18.01 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 21.82 | 18.26 | 18.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 22.51 | 18.34 | 18.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 19.50 | 19.56 | 19.02 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 17.86 | 18.70 | 18.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 17.60 | 18.68 | 18.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 17.73 | 17.67 | 18.10 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 18.85 | 18.39 | 18.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 19.26 | 18.40 | 18.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 18.61 | 18.62 | 18.51 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 14:15:00 | 17.70 | 18.43 | 18.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 17.41 | 18.32 | 18.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 14.67 | 14.08 | 15.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 13.75 | 14.40 | 14.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 14.67 | 14.41 | 14.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 10:15:00 | 15.76 | 14.42 | 14.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 15.61 | 14.86 | 14.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 15.87 | 14.96 | 14.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 20.54 | 20.69 | 18.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 13:15:00 | 21.43 | 20.67 | 19.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 19.36 | 20.66 | 19.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 15:15:00 | 19.59 | 20.65 | 19.29 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 19.22 | 20.63 | 19.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 18.09 | 18.97 | 18.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 17.86 | 18.93 | 18.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 12:15:00 | 18.60 | 18.57 | 18.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-06 10:15:00 | 18.18 | 18.57 | 18.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 18.70 | 18.44 | 18.66 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 20.04 | 18.46 | 18.45 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 17.73 | 18.56 | 18.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 17.62 | 18.53 | 18.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 18.20 | 18.05 | 18.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 17.28 | 17.91 | 18.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 15.12 | 14.48 | 15.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-19 11:15:00 | 15.81 | 14.55 | 15.31 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 19.26 | 15.61 | 15.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 11:15:00 | 19.60 | 15.65 | 15.62 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-27 10:15:00 | 18.14 | 2024-09-19 11:15:00 | 16.76 | TARGET | 1.38 |
| SELL | 2024-09-24 12:15:00 | 17.72 | 2024-09-30 11:15:00 | 18.26 | EXIT_EMA400 | -0.54 |
| SELL | 2025-04-07 09:15:00 | 13.75 | 2025-04-15 10:15:00 | 15.76 | EXIT_EMA400 | -2.01 |
| BUY | 2025-07-29 13:15:00 | 21.43 | 2025-08-05 09:15:00 | 19.22 | EXIT_EMA400 | -2.21 |
| BUY | 2025-08-04 15:15:00 | 19.59 | 2025-08-05 09:15:00 | 19.22 | EXIT_EMA400 | -0.37 |
| SELL | 2025-10-06 10:15:00 | 18.18 | 2025-10-10 09:15:00 | 18.70 | EXIT_EMA400 | -0.52 |
| SELL | 2026-01-08 09:15:00 | 17.28 | 2026-01-23 15:15:00 | 14.62 | TARGET | 2.66 |
