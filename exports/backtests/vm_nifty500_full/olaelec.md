# Ola Electric Mobility Ltd. (OLAELEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-08-09 09:15:00 → 2026-04-30 15:30:00 (2960 bars)
- **Last close:** 36.56
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 27.21
- **Avg P&L per closed trade:** 5.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 95.56 | 89.30 | 89.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 96.24 | 89.49 | 89.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 90.97 | 91.04 | 90.25 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 83.09 | 89.58 | 89.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 82.62 | 89.44 | 89.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 54.14 | 53.52 | 58.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-24 15:15:00 | 52.41 | 53.48 | 58.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 54.30 | 51.29 | 54.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-26 14:15:00 | 52.55 | 51.42 | 54.40 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 53.98 | 51.64 | 54.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-03 09:15:00 | 50.50 | 51.67 | 54.03 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-14 15:15:00 | 47.66 | 44.26 | 47.49 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 54.43 | 45.59 | 45.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 57.89 | 45.71 | 45.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 55.22 | 55.28 | 52.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 10:15:00 | 56.78 | 55.18 | 52.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-06 09:15:00 | 52.29 | 55.22 | 52.64 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 45.49 | 51.91 | 51.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 45.00 | 51.78 | 51.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 15:15:00 | 37.60 | 37.59 | 41.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 12:15:00 | 37.34 | 38.91 | 40.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 09:15:00 | 29.48 | 25.65 | 28.71 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 40.43 | 30.74 | 30.72 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-05-26 14:15:00 | 52.55 | 2025-06-13 09:15:00 | 47.01 | TARGET | 5.54 |
| SELL | 2025-06-03 09:15:00 | 50.50 | 2025-07-10 15:15:00 | 39.90 | TARGET | 10.60 |
| SELL | 2025-04-24 15:15:00 | 52.41 | 2025-07-14 15:15:00 | 47.66 | EXIT_EMA400 | 4.75 |
| BUY | 2025-09-30 10:15:00 | 56.78 | 2025-10-06 09:15:00 | 52.29 | EXIT_EMA400 | -4.49 |
| SELL | 2026-01-16 12:15:00 | 37.34 | 2026-02-20 14:15:00 | 26.52 | TARGET | 10.82 |
