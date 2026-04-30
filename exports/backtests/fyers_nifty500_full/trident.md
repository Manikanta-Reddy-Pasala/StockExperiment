# Trident Ltd. (TRIDENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 25.98
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 5.98
- **Avg P&L per closed trade:** 1.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 36.89 | 37.99 | 38.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 36.44 | 37.51 | 37.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 36.92 | 36.68 | 37.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-04 15:15:00 | 35.55 | 36.61 | 37.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 36.49 | 35.93 | 36.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-17 09:15:00 | 35.68 | 35.93 | 36.54 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 33.81 | 33.20 | 34.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 34.48 | 33.26 | 34.23 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 35.22 | 34.84 | 34.83 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 34.52 | 34.83 | 34.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 34.25 | 34.83 | 34.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 34.32 | 34.25 | 34.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-03 14:15:00 | 33.99 | 34.25 | 34.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 27.30 | 26.18 | 27.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 11:15:00 | 27.45 | 26.20 | 27.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 15:15:00 | 29.53 | 27.82 | 27.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 33.48 | 27.95 | 27.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 30.26 | 30.45 | 29.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 30.97 | 30.45 | 29.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 29.66 | 30.43 | 29.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 29.32 | 30.42 | 29.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 27.96 | 30.29 | 30.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 27.85 | 30.20 | 30.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 29.45 | 28.67 | 29.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-29 14:15:00 | 28.15 | 29.13 | 29.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 29.03 | 28.50 | 28.84 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 09:15:00 | 35.68 | 2024-10-22 15:15:00 | 33.11 | TARGET | 2.57 |
| SELL | 2024-10-04 15:15:00 | 35.55 | 2024-11-22 09:15:00 | 31.11 | TARGET | 4.44 |
| SELL | 2025-01-03 14:15:00 | 33.99 | 2025-01-06 15:15:00 | 32.49 | TARGET | 1.50 |
| BUY | 2025-06-17 09:15:00 | 30.97 | 2025-06-19 12:15:00 | 29.32 | EXIT_EMA400 | -1.65 |
| SELL | 2025-09-29 14:15:00 | 28.15 | 2025-10-23 09:15:00 | 29.03 | EXIT_EMA400 | -0.88 |
