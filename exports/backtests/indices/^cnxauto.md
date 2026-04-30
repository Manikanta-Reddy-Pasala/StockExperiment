# NIFTY AUTO (^CNXAUTO)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5015 bars)
- **Last close:** 25917.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 5000 pts (index)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -643.65
- **Avg P&L per closed trade:** -160.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 23785.15 | 25639.34 | 25642.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 23527.25 | 25493.87 | 25568.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 23892.15 | 23848.52 | 24360.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-10 09:15:00 | 23728.70 | 23849.60 | 24328.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 23814.00 | 23361.20 | 23824.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 13:15:00 | 23981.55 | 23367.38 | 23824.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 23020.55 | 21874.71 | 21873.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 23124.10 | 21978.94 | 21927.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 23190.85 | 23266.18 | 22854.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 12:15:00 | 23204.50 | 23264.02 | 22859.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 23539.55 | 23783.75 | 23521.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 10:15:00 | 23682.30 | 23782.74 | 23522.76 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 23498.10 | 23775.05 | 23526.56 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 26505.95 | 27358.63 | 27361.40 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 27718.25 | 27365.32 | 27364.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 28031.20 | 27437.36 | 27402.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 27729.75 | 27771.40 | 27602.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-23 09:15:00 | 28070.25 | 27777.32 | 27613.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 27793.70 | 27920.99 | 27716.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 10:15:00 | 27570.05 | 27917.50 | 27715.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 26639.80 | 27555.01 | 27555.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 26331.10 | 27517.88 | 27536.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 25964.80 | 25428.86 | 26206.37 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-10 09:15:00 | 23728.70 | 2025-01-02 13:15:00 | 23981.55 | EXIT_EMA400 | -252.85 |
| BUY | 2025-06-13 12:15:00 | 23204.50 | 2025-08-01 09:15:00 | 23498.10 | EXIT_EMA400 | 293.60 |
| BUY | 2025-07-31 10:15:00 | 23682.30 | 2025-08-01 09:15:00 | 23498.10 | EXIT_EMA400 | -184.20 |
| BUY | 2026-02-23 09:15:00 | 28070.25 | 2026-03-02 10:15:00 | 27570.05 | EXIT_EMA400 | -500.20 |
