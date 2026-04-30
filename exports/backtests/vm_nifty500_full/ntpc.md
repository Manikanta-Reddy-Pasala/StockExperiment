# NTPC Ltd. (NTPC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 399.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 42.51
- **Avg P&L per closed trade:** 5.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 385.85 | 408.34 | 408.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 381.65 | 407.39 | 407.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 335.75 | 335.74 | 353.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 09:15:00 | 331.90 | 335.69 | 352.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 324.75 | 317.68 | 328.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-06 11:15:00 | 330.15 | 317.91 | 328.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 11:15:00 | 360.25 | 333.46 | 333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 364.00 | 335.92 | 334.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 12:15:00 | 345.25 | 341.45 | 337.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 347.15 | 352.82 | 346.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-06 09:15:00 | 344.05 | 352.50 | 346.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 10:15:00 | 337.40 | 339.71 | 341.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 339.45 | 336.01 | 338.74 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.25 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.44 | 336.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 339.80 | 337.12 | 336.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 339.80 | 337.12 | 336.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-10 10:15:00 | 340.75 | 337.15 | 336.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 337.90 | 337.30 | 336.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 10:15:00 | 338.60 | 337.31 | 336.93 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 12:15:00 | 335.65 | 337.50 | 337.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 325.95 | 337.52 | 337.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 325.20 | 336.74 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.40 | 324.54 | 328.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 13:15:00 | 323.40 | 324.53 | 328.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.28 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.60 | 334.41 | 332.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.35 | 336.97 | 334.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-27 09:15:00 | 342.40 | 337.02 | 334.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.20 | 374.15 | 365.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-01 13:15:00 | 365.40 | 374.00 | 365.53 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-21 09:15:00 | 331.90 | 2025-03-06 11:15:00 | 330.15 | EXIT_EMA400 | 1.75 |
| BUY | 2025-04-07 12:15:00 | 345.25 | 2025-04-15 12:15:00 | 367.30 | TARGET | 22.05 |
| SELL | 2025-06-12 10:15:00 | 337.40 | 2025-06-24 09:15:00 | 325.11 | TARGET | 12.29 |
| BUY | 2025-10-10 09:15:00 | 339.80 | 2025-10-14 12:15:00 | 335.65 | EXIT_EMA400 | -4.15 |
| BUY | 2025-10-10 10:15:00 | 340.75 | 2025-10-14 12:15:00 | 335.65 | EXIT_EMA400 | -5.10 |
| BUY | 2025-10-13 10:15:00 | 338.60 | 2025-10-14 12:15:00 | 335.65 | EXIT_EMA400 | -2.95 |
| SELL | 2025-12-24 13:15:00 | 323.40 | 2025-12-31 09:15:00 | 328.90 | EXIT_EMA400 | -5.50 |
| BUY | 2026-01-27 09:15:00 | 342.40 | 2026-02-04 09:15:00 | 366.52 | TARGET | 24.12 |
