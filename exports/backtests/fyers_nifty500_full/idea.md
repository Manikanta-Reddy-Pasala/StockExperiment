# Vodafone Idea Ltd. (IDEA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 10.19
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -2.51
- **Avg P&L per closed trade:** -0.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 13.43 | 15.60 | 15.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 13.22 | 15.55 | 15.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 8.26 | 8.20 | 9.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 13:15:00 | 7.54 | 8.19 | 9.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 8.42 | 7.95 | 8.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-15 10:15:00 | 8.57 | 7.96 | 8.46 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 9.32 | 8.76 | 8.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 9.41 | 8.79 | 8.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 10:15:00 | 8.92 | 8.93 | 8.85 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 8.07 | 8.79 | 8.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 8.04 | 8.78 | 8.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 10:15:00 | 7.82 | 7.55 | 7.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 7.25 | 7.67 | 7.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 7.97 | 7.52 | 7.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 7.60 | 7.21 | 7.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 7.83 | 7.21 | 7.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 7.33 | 7.35 | 7.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 7.43 | 7.35 | 7.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 7.43 | 7.35 | 7.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 10:15:00 | 7.29 | 7.35 | 7.29 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 6.77 | 7.24 | 7.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 6.58 | 7.11 | 7.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 7.20 | 6.86 | 7.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 6.64 | 6.89 | 7.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 6.64 | 6.89 | 7.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 09:15:00 | 6.57 | 6.88 | 7.01 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6.76 | 6.79 | 6.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-04 14:15:00 | 6.61 | 6.78 | 6.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-05 12:15:00 | 6.97 | 6.78 | 6.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 8.11 | 7.04 | 7.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 8.58 | 7.24 | 7.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 8.39 | 8.72 | 8.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 14:15:00 | 9.54 | 8.74 | 8.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 10.71 | 11.12 | 10.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-01 09:15:00 | 11.29 | 11.12 | 10.35 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 10.65 | 11.22 | 10.66 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 10.05 | 10.73 | 10.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 9.94 | 10.71 | 10.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 9.45 | 9.41 | 9.85 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 13:15:00 | 7.54 | 2025-01-15 10:15:00 | 8.57 | EXIT_EMA400 | -1.03 |
| SELL | 2025-04-07 09:15:00 | 7.25 | 2025-04-21 13:15:00 | 7.97 | EXIT_EMA400 | -0.72 |
| BUY | 2025-07-28 09:15:00 | 7.43 | 2025-07-28 10:15:00 | 7.29 | EXIT_EMA400 | -0.14 |
| SELL | 2025-08-26 09:15:00 | 6.64 | 2025-09-05 12:15:00 | 6.97 | EXIT_EMA400 | -0.33 |
| SELL | 2025-08-28 09:15:00 | 6.57 | 2025-09-05 12:15:00 | 6.97 | EXIT_EMA400 | -0.40 |
| SELL | 2025-09-04 14:15:00 | 6.61 | 2025-09-05 12:15:00 | 6.97 | EXIT_EMA400 | -0.36 |
| BUY | 2025-11-03 14:15:00 | 9.54 | 2026-01-19 09:15:00 | 10.65 | EXIT_EMA400 | 1.11 |
| BUY | 2026-01-01 09:15:00 | 11.29 | 2026-01-19 09:15:00 | 10.65 | EXIT_EMA400 | -0.64 |
