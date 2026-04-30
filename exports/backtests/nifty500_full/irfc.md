# Indian Railway Finance Corporation Ltd. (IRFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 104.21
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 23.88
- **Avg P&L per closed trade:** 4.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 168.82 | 180.07 | 180.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 167.93 | 179.05 | 179.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 152.20 | 150.79 | 159.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 147.67 | 151.65 | 157.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 154.85 | 148.02 | 153.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 136.60 | 129.28 | 129.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 137.02 | 129.50 | 129.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 138.25 | 138.35 | 134.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 139.98 | 138.37 | 135.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 135.48 | 138.27 | 135.27 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 134.74 | 138.24 | 135.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 10:15:00 | 131.52 | 135.66 | 135.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 128.93 | 135.42 | 135.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 126.49 | 125.80 | 128.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 11:15:00 | 125.20 | 126.94 | 128.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 124.77 | 125.78 | 127.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-16 13:15:00 | 124.52 | 125.71 | 127.09 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 119.70 | 115.98 | 119.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 127.74 | 121.19 | 121.19 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 115.31 | 121.31 | 121.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 113.53 | 120.29 | 120.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 120.10 | 119.87 | 120.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 114.21 | 119.82 | 120.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 102.78 | 98.49 | 103.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 104.42 | 98.55 | 103.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 15:15:00 | 147.67 | 2024-11-28 09:15:00 | 154.85 | EXIT_EMA400 | -7.18 |
| BUY | 2025-06-16 13:15:00 | 139.98 | 2025-06-19 10:15:00 | 134.74 | EXIT_EMA400 | -5.24 |
| SELL | 2025-10-16 13:15:00 | 124.52 | 2025-11-24 14:15:00 | 116.82 | TARGET | 7.70 |
| SELL | 2025-09-25 11:15:00 | 125.20 | 2025-12-03 09:15:00 | 115.36 | TARGET | 9.84 |
| SELL | 2026-02-02 09:15:00 | 114.21 | 2026-03-09 09:15:00 | 95.45 | TARGET | 18.76 |
