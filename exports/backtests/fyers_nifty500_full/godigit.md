# Go Digit General Insurance Ltd. (GODIGIT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-23 09:15:00 → 2026-04-30 15:15:00 (3357 bars)
- **Last close:** 307.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -29.30
- **Avg P&L per closed trade:** -7.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 336.50 | 359.13 | 359.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 332.30 | 358.86 | 359.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 11:15:00 | 340.95 | 338.75 | 346.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 325.45 | 338.62 | 346.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 14:15:00 | 348.00 | 338.67 | 346.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 322.50 | 298.83 | 298.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 326.85 | 300.80 | 299.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 335.80 | 339.24 | 327.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 11:15:00 | 346.00 | 338.55 | 329.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 350.95 | 358.99 | 351.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 345.40 | 353.77 | 353.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 340.70 | 352.89 | 353.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 349.95 | 349.40 | 351.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 11:15:00 | 346.15 | 349.33 | 351.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 344.10 | 347.36 | 349.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-30 10:15:00 | 341.15 | 346.80 | 349.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 348.30 | 346.41 | 348.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 349.50 | 346.44 | 348.75 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 12:15:00 | 325.45 | 2024-11-25 14:15:00 | 348.00 | EXIT_EMA400 | -22.55 |
| BUY | 2025-07-14 11:15:00 | 346.00 | 2025-09-02 13:15:00 | 350.95 | EXIT_EMA400 | 4.95 |
| SELL | 2025-12-16 11:15:00 | 346.15 | 2026-01-05 11:15:00 | 349.50 | EXIT_EMA400 | -3.35 |
| SELL | 2025-12-30 10:15:00 | 341.15 | 2026-01-05 11:15:00 | 349.50 | EXIT_EMA400 | -8.35 |
