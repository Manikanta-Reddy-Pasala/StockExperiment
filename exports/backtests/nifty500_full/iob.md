# Indian Overseas Bank (IOB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 35.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 7.49
- **Avg P&L per closed trade:** 1.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 10:15:00 | 62.14 | 64.88 | 64.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 61.67 | 64.81 | 64.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 60.37 | 59.98 | 61.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 11:15:00 | 58.52 | 59.98 | 61.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 54.50 | 52.93 | 54.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 13:15:00 | 54.40 | 52.95 | 54.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 53.95 | 52.99 | 54.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-29 10:15:00 | 53.40 | 53.00 | 54.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-03 10:15:00 | 54.84 | 53.06 | 54.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 39.90 | 38.49 | 38.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 40.20 | 38.62 | 38.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 38.92 | 39.21 | 38.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 11:15:00 | 39.47 | 39.10 | 38.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 39.12 | 39.24 | 38.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-08 12:15:00 | 39.21 | 39.24 | 38.99 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-09 09:15:00 | 38.86 | 39.23 | 38.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 38.10 | 39.30 | 39.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 37.91 | 39.23 | 39.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 36.33 | 36.30 | 37.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 35.88 | 36.41 | 37.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 35.83 | 35.31 | 35.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-17 11:15:00 | 36.05 | 35.32 | 35.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-25 11:15:00 | 58.52 | 2024-10-22 15:15:00 | 49.95 | TARGET | 8.57 |
| SELL | 2024-11-28 13:15:00 | 54.40 | 2024-11-29 11:15:00 | 52.91 | TARGET | 1.49 |
| SELL | 2024-11-29 10:15:00 | 53.40 | 2024-12-03 10:15:00 | 54.84 | EXIT_EMA400 | -1.44 |
| BUY | 2025-09-30 11:15:00 | 39.47 | 2025-10-09 09:15:00 | 38.86 | EXIT_EMA400 | -0.61 |
| BUY | 2025-10-08 12:15:00 | 39.21 | 2025-10-09 09:15:00 | 38.86 | EXIT_EMA400 | -0.35 |
| SELL | 2026-01-08 10:15:00 | 35.88 | 2026-02-17 11:15:00 | 36.05 | EXIT_EMA400 | -0.17 |
