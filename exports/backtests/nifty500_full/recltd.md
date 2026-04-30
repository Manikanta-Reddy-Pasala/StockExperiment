# REC Ltd. (RECLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 354.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 4 / 1
- **Total realized P&L (per unit):** 134.52
- **Avg P&L per closed trade:** 26.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 548.15 | 576.21 | 576.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 543.00 | 575.09 | 575.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 554.35 | 552.43 | 561.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 544.95 | 552.35 | 561.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 531.00 | 523.32 | 535.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 12:15:00 | 529.00 | 523.46 | 535.43 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 533.90 | 523.94 | 535.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 535.20 | 524.22 | 535.18 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 555.90 | 541.91 | 541.89 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 538.55 | 541.87 | 541.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 526.15 | 541.62 | 541.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 532.45 | 526.33 | 532.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 518.60 | 526.85 | 532.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 425.65 | 411.86 | 435.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 435.25 | 414.35 | 435.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 374.70 | 363.23 | 363.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 375.30 | 363.71 | 363.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 365.05 | 365.09 | 364.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 368.55 | 364.73 | 364.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-09 09:15:00 | 357.10 | 367.35 | 365.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 343.45 | 364.06 | 364.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 338.80 | 358.07 | 360.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 346.35 | 345.72 | 352.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 340.40 | 345.72 | 352.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 342.65 | 334.03 | 342.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 15:15:00 | 343.20 | 334.12 | 342.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 379.40 | 348.78 | 348.73 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 09:15:00 | 544.95 | 2024-10-23 09:15:00 | 496.05 | TARGET | 48.90 |
| SELL | 2024-11-28 12:15:00 | 529.00 | 2024-12-02 09:15:00 | 535.20 | EXIT_EMA400 | -6.20 |
| SELL | 2025-01-06 10:15:00 | 518.60 | 2025-01-10 09:15:00 | 475.66 | TARGET | 42.94 |
| BUY | 2026-02-03 09:15:00 | 368.55 | 2026-02-04 14:15:00 | 381.97 | TARGET | 13.42 |
| SELL | 2026-03-19 09:15:00 | 340.40 | 2026-03-30 14:15:00 | 304.93 | TARGET | 35.47 |
