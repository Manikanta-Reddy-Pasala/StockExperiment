# Jaiprakash Power Ventures Ltd. (JPPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 19.72
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -2.40
- **Avg P&L per closed trade:** -0.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 11:15:00 | 15.40 | 16.78 | 16.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 15.30 | 16.73 | 16.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 16.80 | 16.67 | 16.72 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 10:15:00 | 18.45 | 16.78 | 16.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 18.90 | 16.88 | 16.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 14:15:00 | 17.20 | 17.29 | 17.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-18 11:15:00 | 17.65 | 17.29 | 17.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-19 09:15:00 | 16.95 | 17.29 | 17.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 17.84 | 18.77 | 18.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 17.55 | 18.71 | 18.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 11:15:00 | 18.61 | 18.48 | 18.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 10:15:00 | 18.14 | 18.48 | 18.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 17.94 | 17.74 | 18.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-24 12:15:00 | 17.72 | 17.74 | 18.07 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 17.45 | 17.71 | 18.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-30 11:15:00 | 18.28 | 17.71 | 18.01 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 21.85 | 18.26 | 18.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 22.52 | 18.34 | 18.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 19.49 | 19.56 | 19.02 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 17.93 | 18.70 | 18.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 17.60 | 18.69 | 18.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 17.74 | 17.67 | 18.10 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 18.85 | 18.39 | 18.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 19.26 | 18.40 | 18.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 18.60 | 18.62 | 18.51 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 14:15:00 | 17.70 | 18.43 | 18.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 17.41 | 18.32 | 18.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 14.66 | 14.08 | 15.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 13.75 | 14.40 | 14.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 14.70 | 14.41 | 14.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 10:15:00 | 15.78 | 14.42 | 14.84 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 15.63 | 14.86 | 14.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 15:15:00 | 15.70 | 14.87 | 14.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 20.54 | 20.69 | 18.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 13:15:00 | 21.43 | 20.67 | 19.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 19.36 | 20.66 | 19.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 15:15:00 | 19.59 | 20.65 | 19.29 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 19.21 | 20.63 | 19.29 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 18.04 | 18.98 | 18.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 17.85 | 18.93 | 18.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 12:15:00 | 18.60 | 18.57 | 18.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-06 10:15:00 | 18.18 | 18.57 | 18.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 18.70 | 18.44 | 18.66 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 20.04 | 18.46 | 18.45 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 17.73 | 18.56 | 18.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 17.62 | 18.53 | 18.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 18.20 | 18.05 | 18.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 17.29 | 17.91 | 18.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 15.12 | 14.49 | 15.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-19 11:15:00 | 15.81 | 14.55 | 15.33 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 19.26 | 15.61 | 15.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 11:15:00 | 19.60 | 15.65 | 15.63 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-18 11:15:00 | 17.65 | 2024-04-19 09:15:00 | 16.95 | EXIT_EMA400 | -0.70 |
| SELL | 2024-08-27 10:15:00 | 18.14 | 2024-09-19 11:15:00 | 16.76 | TARGET | 1.38 |
| SELL | 2024-09-24 12:15:00 | 17.72 | 2024-09-30 11:15:00 | 18.28 | EXIT_EMA400 | -0.56 |
| SELL | 2025-04-07 09:15:00 | 13.75 | 2025-04-15 10:15:00 | 15.78 | EXIT_EMA400 | -2.03 |
| BUY | 2025-07-29 13:15:00 | 21.43 | 2025-08-05 09:15:00 | 19.21 | EXIT_EMA400 | -2.22 |
| BUY | 2025-08-04 15:15:00 | 19.59 | 2025-08-05 09:15:00 | 19.21 | EXIT_EMA400 | -0.38 |
| SELL | 2025-10-06 10:15:00 | 18.18 | 2025-10-10 09:15:00 | 18.70 | EXIT_EMA400 | -0.52 |
| SELL | 2026-01-08 09:15:00 | 17.29 | 2026-01-27 09:15:00 | 14.66 | TARGET | 2.63 |
