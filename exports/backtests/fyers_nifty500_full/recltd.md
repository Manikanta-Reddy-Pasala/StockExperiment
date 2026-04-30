# REC Ltd. (RECLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 355.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 3 / 1
- **Total realized P&L (per unit):** 87.11
- **Avg P&L per closed trade:** 21.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 538.60 | 579.57 | 579.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 529.95 | 566.54 | 572.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 553.95 | 552.46 | 562.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 544.95 | 552.38 | 562.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 530.95 | 523.27 | 535.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 12:15:00 | 529.00 | 523.41 | 535.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 535.20 | 524.17 | 535.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 535.60 | 524.77 | 535.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 374.70 | 363.20 | 363.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 375.30 | 363.70 | 363.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 361.25 | 365.05 | 364.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-01 11:15:00 | 366.40 | 365.04 | 364.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 371.15 | 365.10 | 364.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-01 13:15:00 | 364.10 | 365.09 | 364.24 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 343.55 | 364.04 | 364.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 338.80 | 358.06 | 360.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 346.35 | 345.71 | 352.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 340.40 | 345.71 | 352.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 338.55 | 333.64 | 342.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 15:15:00 | 343.10 | 334.12 | 342.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 384.00 | 348.81 | 348.74 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 09:15:00 | 544.95 | 2024-11-21 09:15:00 | 493.27 | TARGET | 51.68 |
| SELL | 2024-11-28 12:15:00 | 529.00 | 2024-12-03 09:15:00 | 535.60 | EXIT_EMA400 | -6.60 |
| BUY | 2026-02-01 11:15:00 | 366.40 | 2026-02-01 12:15:00 | 372.97 | TARGET | 6.57 |
| SELL | 2026-03-19 09:15:00 | 340.40 | 2026-03-30 14:15:00 | 304.94 | TARGET | 35.46 |
