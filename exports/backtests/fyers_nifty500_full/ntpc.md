# NTPC Ltd. (NTPC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 400.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 7.33
- **Avg P&L per closed trade:** 0.81

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 383.65 | 408.52 | 408.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 377.70 | 406.56 | 407.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 14:15:00 | 336.15 | 335.75 | 353.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 09:15:00 | 331.90 | 335.70 | 352.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 324.70 | 317.58 | 327.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-06 11:15:00 | 330.15 | 317.80 | 327.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 362.50 | 332.88 | 332.85 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 328.85 | 343.21 | 343.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 10:15:00 | 337.35 | 339.70 | 341.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 339.45 | 336.02 | 338.72 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 340.95 | 339.47 | 339.47 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 337.15 | 339.46 | 339.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 335.65 | 339.40 | 339.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-01 09:15:00 | 332.35 | 338.32 | 338.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 336.85 | 336.32 | 337.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-11 12:15:00 | 334.65 | 336.28 | 337.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 336.00 | 336.27 | 337.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-12 09:15:00 | 339.20 | 336.30 | 337.56 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.10 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.46 | 336.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 339.95 | 337.14 | 336.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 339.95 | 337.14 | 336.83 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-10 10:15:00 | 340.75 | 337.18 | 336.85 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 338.00 | 337.32 | 336.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 12:15:00 | 339.10 | 337.37 | 336.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 12:15:00 | 335.60 | 337.52 | 337.05 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 326.00 | 337.52 | 337.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 325.10 | 336.73 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.45 | 324.54 | 328.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 14:15:00 | 322.40 | 324.51 | 328.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.27 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.50 | 334.39 | 332.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.30 | 336.95 | 334.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-27 09:15:00 | 342.40 | 337.01 | 334.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.15 | 374.18 | 365.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-01 13:15:00 | 365.40 | 374.03 | 365.62 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-21 09:15:00 | 331.90 | 2025-03-06 11:15:00 | 330.15 | EXIT_EMA400 | 1.75 |
| SELL | 2025-06-12 10:15:00 | 337.35 | 2025-06-24 09:15:00 | 325.02 | TARGET | 12.33 |
| SELL | 2025-08-01 09:15:00 | 332.35 | 2025-08-12 09:15:00 | 339.20 | EXIT_EMA400 | -6.85 |
| SELL | 2025-08-11 12:15:00 | 334.65 | 2025-08-12 09:15:00 | 339.20 | EXIT_EMA400 | -4.55 |
| BUY | 2025-10-10 09:15:00 | 339.95 | 2025-10-14 12:15:00 | 335.60 | EXIT_EMA400 | -4.35 |
| BUY | 2025-10-10 10:15:00 | 340.75 | 2025-10-14 12:15:00 | 335.60 | EXIT_EMA400 | -5.15 |
| BUY | 2025-10-13 12:15:00 | 339.10 | 2025-10-14 12:15:00 | 335.60 | EXIT_EMA400 | -3.50 |
| SELL | 2025-12-24 14:15:00 | 322.40 | 2025-12-31 09:15:00 | 328.90 | EXIT_EMA400 | -6.50 |
| BUY | 2026-01-27 09:15:00 | 342.40 | 2026-02-04 09:15:00 | 366.55 | TARGET | 24.15 |
