# Indian Renewable Energy Development Agency Ltd. (IREDA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 135.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 29.16
- **Avg P&L per closed trade:** 5.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 213.25 | 230.60 | 230.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 203.40 | 226.12 | 228.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 207.61 | 201.17 | 210.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-26 11:15:00 | 199.35 | 208.43 | 210.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 14:15:00 | 218.87 | 207.24 | 210.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 219.27 | 212.30 | 212.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 222.82 | 212.41 | 212.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 211.50 | 213.15 | 212.73 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 200.87 | 212.34 | 212.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 199.55 | 212.21 | 212.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 206.89 | 203.25 | 206.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 11:15:00 | 202.07 | 203.26 | 206.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 202.07 | 203.26 | 206.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-01 12:15:00 | 194.95 | 203.18 | 206.83 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 172.20 | 160.24 | 173.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 171.70 | 160.35 | 173.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 15:15:00 | 168.16 | 159.15 | 168.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 175.70 | 159.31 | 168.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 175.49 | 169.87 | 169.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 177.19 | 170.50 | 170.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 165.85 | 170.62 | 170.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 161.07 | 169.00 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 148.14 | 147.62 | 152.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-11 12:15:00 | 146.71 | 147.64 | 152.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 153.82 | 147.66 | 152.28 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 137.81 | 125.76 | 125.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 139.34 | 126.02 | 125.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-26 11:15:00 | 199.35 | 2024-12-30 14:15:00 | 218.87 | EXIT_EMA400 | -19.52 |
| SELL | 2025-02-01 11:15:00 | 202.07 | 2025-02-03 09:15:00 | 187.62 | TARGET | 14.45 |
| SELL | 2025-02-01 12:15:00 | 194.95 | 2025-02-18 13:15:00 | 159.32 | TARGET | 35.63 |
| SELL | 2025-03-25 10:15:00 | 171.70 | 2025-03-26 13:15:00 | 165.98 | TARGET | 5.72 |
| SELL | 2025-09-11 12:15:00 | 146.71 | 2025-09-15 09:15:00 | 153.82 | EXIT_EMA400 | -7.11 |
