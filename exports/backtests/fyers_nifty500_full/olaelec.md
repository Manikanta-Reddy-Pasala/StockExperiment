# Ola Electric Mobility Ltd. (OLAELEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-08-09 09:15:00 → 2026-04-30 15:15:00 (2979 bars)
- **Last close:** 36.55
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
- **Target hits / EMA400 exits:** 4 / 1
- **Total realized P&L (per unit):** 41.24
- **Avg P&L per closed trade:** 8.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 95.62 | 89.22 | 89.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 96.24 | 89.47 | 89.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 90.96 | 91.02 | 90.23 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 82.79 | 89.50 | 89.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 82.62 | 89.44 | 89.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 78.35 | 76.51 | 81.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 14:15:00 | 74.66 | 76.51 | 80.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 54.30 | 51.28 | 54.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-26 14:15:00 | 52.55 | 51.42 | 54.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 53.95 | 51.64 | 54.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-03 09:15:00 | 50.50 | 51.67 | 54.02 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-14 15:15:00 | 47.71 | 44.26 | 47.49 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 54.42 | 45.59 | 45.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 57.87 | 45.71 | 45.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 55.22 | 55.28 | 52.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 10:15:00 | 56.86 | 55.18 | 52.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-06 09:15:00 | 52.29 | 55.22 | 52.64 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 45.49 | 51.91 | 51.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 44.22 | 51.70 | 51.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 15:15:00 | 37.65 | 37.59 | 41.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 12:15:00 | 37.32 | 38.90 | 40.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 09:15:00 | 29.48 | 25.63 | 28.65 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 40.28 | 30.73 | 30.68 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-01 14:15:00 | 74.66 | 2025-02-28 09:15:00 | 55.78 | TARGET | 18.88 |
| SELL | 2025-05-26 14:15:00 | 52.55 | 2025-06-13 09:15:00 | 47.05 | TARGET | 5.50 |
| SELL | 2025-06-03 09:15:00 | 50.50 | 2025-07-10 10:15:00 | 39.94 | TARGET | 10.56 |
| BUY | 2025-09-30 10:15:00 | 56.86 | 2025-10-06 09:15:00 | 52.29 | EXIT_EMA400 | -4.57 |
| SELL | 2026-01-16 12:15:00 | 37.32 | 2026-02-20 15:15:00 | 26.45 | TARGET | 10.87 |
