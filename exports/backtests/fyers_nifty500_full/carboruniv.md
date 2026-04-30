# Carborundum Universal Ltd. (CARBORUNIV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 958.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 84.62
- **Avg P&L per closed trade:** 28.21

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 1594.75 | 1643.57 | 1643.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 11:15:00 | 1584.85 | 1642.98 | 1643.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1522.50 | 1493.45 | 1530.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 11:15:00 | 1464.50 | 1494.21 | 1528.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-18 09:15:00 | 1499.50 | 1435.57 | 1471.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1002.10 | 949.82 | 949.65 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 925.00 | 950.64 | 950.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 916.25 | 950.30 | 950.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 09:15:00 | 923.15 | 937.93 | 943.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 932.85 | 923.02 | 932.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 09:15:00 | 912.95 | 923.06 | 932.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-26 11:15:00 | 909.10 | 883.76 | 904.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 892.50 | 818.14 | 818.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 907.70 | 826.18 | 822.24 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 11:15:00 | 1464.50 | 2024-11-18 09:15:00 | 1499.50 | EXIT_EMA400 | -35.00 |
| SELL | 2025-10-13 09:15:00 | 923.15 | 2025-11-18 09:15:00 | 862.22 | TARGET | 60.93 |
| SELL | 2025-10-31 09:15:00 | 912.95 | 2025-11-18 14:15:00 | 854.26 | TARGET | 58.69 |
