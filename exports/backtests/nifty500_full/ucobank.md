# UCO Bank (UCOBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 26.79
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 5.57
- **Avg P&L per closed trade:** 1.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 12:15:00 | 51.22 | 55.20 | 55.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 50.86 | 54.64 | 54.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 13:15:00 | 50.36 | 50.24 | 51.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-24 09:15:00 | 49.37 | 50.23 | 51.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 48.16 | 44.55 | 46.01 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 31.83 | 30.16 | 30.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 32.32 | 30.60 | 30.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 31.55 | 31.68 | 31.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-07 10:15:00 | 31.90 | 31.68 | 31.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-12 10:15:00 | 31.15 | 31.73 | 31.19 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 29.42 | 30.99 | 30.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 28.97 | 30.95 | 30.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 29.58 | 29.50 | 30.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 29.32 | 29.63 | 30.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-16 09:15:00 | 29.89 | 29.46 | 29.86 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-24 09:15:00 | 49.37 | 2024-10-23 09:15:00 | 42.48 | TARGET | 6.89 |
| BUY | 2025-11-07 10:15:00 | 31.90 | 2025-11-12 10:15:00 | 31.15 | EXIT_EMA400 | -0.75 |
| SELL | 2026-01-08 10:15:00 | 29.32 | 2026-01-16 09:15:00 | 29.89 | EXIT_EMA400 | -0.57 |
