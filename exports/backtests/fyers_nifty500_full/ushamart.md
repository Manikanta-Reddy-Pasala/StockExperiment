# Usha Martin Ltd. (USHAMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 448.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -114.15
- **Avg P&L per closed trade:** -16.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 344.85 | 367.46 | 367.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 331.00 | 365.71 | 366.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 350.70 | 345.81 | 353.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 09:15:00 | 342.65 | 351.18 | 354.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 350.30 | 350.46 | 354.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-23 13:15:00 | 353.90 | 350.37 | 353.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 422.45 | 355.47 | 355.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 10:15:00 | 428.50 | 357.47 | 356.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 10:15:00 | 383.90 | 384.11 | 372.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-28 09:15:00 | 397.80 | 384.25 | 372.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 386.85 | 398.43 | 385.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-12 14:15:00 | 383.50 | 398.28 | 385.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 374.05 | 385.73 | 385.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 11:15:00 | 372.05 | 385.60 | 385.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 315.00 | 314.35 | 333.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 10:15:00 | 310.85 | 317.60 | 331.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 321.90 | 316.61 | 329.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-19 10:15:00 | 319.00 | 316.63 | 329.64 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 327.10 | 316.98 | 329.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 11:15:00 | 330.10 | 317.22 | 329.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 346.30 | 316.49 | 316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 351.00 | 316.84 | 316.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 360.60 | 362.75 | 347.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 14:15:00 | 366.00 | 362.75 | 347.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 352.10 | 366.00 | 352.97 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 421.40 | 438.71 | 438.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 414.30 | 437.88 | 438.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 413.60 | 424.80 | 429.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.30 | 423.74 | 428.67 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-19 11:15:00 | 413.55 | 423.00 | 427.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 448.80 | 419.91 | 419.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 450.55 | 421.35 | 420.57 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 09:15:00 | 342.65 | 2024-09-23 13:15:00 | 353.90 | EXIT_EMA400 | -11.25 |
| BUY | 2024-10-28 09:15:00 | 397.80 | 2024-11-12 14:15:00 | 383.50 | EXIT_EMA400 | -14.30 |
| SELL | 2025-03-13 10:15:00 | 310.85 | 2025-03-20 11:15:00 | 330.10 | EXIT_EMA400 | -19.25 |
| SELL | 2025-03-19 10:15:00 | 319.00 | 2025-03-20 11:15:00 | 330.10 | EXIT_EMA400 | -11.10 |
| BUY | 2025-07-28 14:15:00 | 366.00 | 2025-08-07 09:15:00 | 352.10 | EXIT_EMA400 | -13.90 |
| SELL | 2026-02-13 09:15:00 | 413.60 | 2026-02-25 09:15:00 | 435.75 | EXIT_EMA400 | -22.15 |
| SELL | 2026-02-19 11:15:00 | 413.55 | 2026-02-25 09:15:00 | 435.75 | EXIT_EMA400 | -22.20 |
