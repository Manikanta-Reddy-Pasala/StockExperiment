# Lemon Tree Hotels Ltd. (LEMONTREE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 117.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -9.47
- **Avg P&L per closed trade:** -1.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 122.08 | 143.66 | 143.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 120.33 | 143.43 | 143.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 135.50 | 133.74 | 137.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-02 09:15:00 | 133.11 | 133.77 | 137.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-11 09:15:00 | 124.20 | 120.07 | 124.05 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 135.45 | 125.39 | 125.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 135.59 | 125.49 | 125.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 145.61 | 146.20 | 139.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-08 15:15:00 | 147.99 | 146.23 | 139.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 139.16 | 146.07 | 139.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 12:15:00 | 127.96 | 138.36 | 138.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 125.74 | 137.95 | 138.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 131.23 | 130.49 | 133.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-28 11:15:00 | 128.27 | 132.69 | 133.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 133.00 | 132.54 | 133.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-01 12:15:00 | 134.47 | 132.56 | 133.71 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 139.70 | 134.66 | 134.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 141.51 | 134.78 | 134.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 137.66 | 138.58 | 136.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-25 15:15:00 | 140.00 | 138.62 | 136.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 139.28 | 138.75 | 137.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-30 12:15:00 | 136.81 | 138.71 | 137.10 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 151.51 | 162.46 | 162.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 150.07 | 160.98 | 161.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 161.28 | 158.87 | 160.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 13:15:00 | 158.75 | 160.81 | 161.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 160.72 | 160.59 | 160.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 10:15:00 | 162.00 | 160.60 | 160.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-02 09:15:00 | 133.11 | 2024-09-30 09:15:00 | 121.11 | TARGET | 12.00 |
| BUY | 2025-01-08 15:15:00 | 147.99 | 2025-01-10 09:15:00 | 139.16 | EXIT_EMA400 | -8.83 |
| SELL | 2025-03-28 11:15:00 | 128.27 | 2025-04-01 12:15:00 | 134.47 | EXIT_EMA400 | -6.20 |
| BUY | 2025-04-25 15:15:00 | 140.00 | 2025-04-30 12:15:00 | 136.81 | EXIT_EMA400 | -3.19 |
| SELL | 2025-12-17 13:15:00 | 158.75 | 2025-12-22 10:15:00 | 162.00 | EXIT_EMA400 | -3.25 |
