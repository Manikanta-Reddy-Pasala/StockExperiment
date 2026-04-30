# Tata Motors Passenger Vehicles Ltd. (TMPV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 342.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 122.33
- **Avg P&L per closed trade:** 17.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 587.85 | 625.09 | 625.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 583.42 | 624.67 | 625.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 495.55 | 495.34 | 524.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 12:15:00 | 488.21 | 495.12 | 523.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 425.97 | 408.80 | 426.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 429.09 | 409.50 | 426.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 442.82 | 412.70 | 412.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 444.94 | 418.24 | 415.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 428.48 | 429.12 | 423.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 433.18 | 429.14 | 423.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 412.85 | 429.01 | 423.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 414.24 | 419.45 | 419.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 413.55 | 418.95 | 419.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 09:15:00 | 405.42 | 416.97 | 417.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 405.42 | 416.97 | 417.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-31 09:15:00 | 403.00 | 416.17 | 417.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-18 09:15:00 | 411.21 | 405.71 | 410.71 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 437.03 | 413.38 | 413.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 437.76 | 419.53 | 417.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 420.39 | 421.42 | 419.02 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 391.60 | 417.05 | 417.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 389.75 | 416.29 | 416.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-04 11:15:00 | 409.30 | 411.48 | 413.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 412.00 | 410.60 | 412.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-10 14:15:00 | 410.25 | 410.60 | 412.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 403.15 | 410.53 | 412.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-12 14:15:00 | 402.05 | 409.86 | 412.21 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 372.85 | 362.74 | 373.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 14:15:00 | 373.80 | 362.94 | 373.46 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 384.50 | 367.53 | 367.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 387.00 | 371.08 | 369.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 373.05 | 373.21 | 370.62 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 331.95 | 368.39 | 368.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 328.90 | 367.64 | 368.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 330.90 | 326.70 | 341.68 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 12:15:00 | 488.21 | 2025-02-28 09:15:00 | 381.91 | TARGET | 106.30 |
| BUY | 2025-06-13 14:15:00 | 433.18 | 2025-06-16 09:15:00 | 412.85 | EXIT_EMA400 | -20.33 |
| SELL | 2025-07-30 09:15:00 | 405.42 | 2025-08-18 09:15:00 | 411.21 | EXIT_EMA400 | -5.79 |
| SELL | 2025-07-31 09:15:00 | 403.00 | 2025-08-18 09:15:00 | 411.21 | EXIT_EMA400 | -8.21 |
| SELL | 2025-11-10 14:15:00 | 410.25 | 2025-11-11 10:15:00 | 402.78 | TARGET | 7.47 |
| SELL | 2025-11-04 11:15:00 | 409.30 | 2025-11-14 09:15:00 | 396.90 | TARGET | 12.40 |
| SELL | 2025-11-12 14:15:00 | 402.05 | 2025-11-17 09:15:00 | 371.56 | TARGET | 30.49 |
