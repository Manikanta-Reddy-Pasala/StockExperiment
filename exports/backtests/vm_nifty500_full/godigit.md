# Go Digit General Insurance Ltd. (GODIGIT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-23 09:15:00 → 2026-04-30 15:30:00 (3338 bars)
- **Last close:** 309.05
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
- **Total realized P&L (per unit):** -30.55
- **Avg P&L per closed trade:** -7.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 336.50 | 359.10 | 359.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 332.25 | 358.84 | 358.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 11:15:00 | 340.95 | 338.86 | 346.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 325.20 | 338.72 | 346.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 14:15:00 | 347.45 | 338.76 | 346.54 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 326.05 | 298.92 | 298.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 326.85 | 300.87 | 299.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 335.80 | 339.23 | 327.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 345.00 | 338.41 | 329.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 350.95 | 359.01 | 351.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 345.40 | 353.76 | 353.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 340.70 | 352.88 | 353.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 349.25 | 348.98 | 350.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 343.60 | 348.91 | 350.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 344.10 | 347.32 | 349.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-30 10:15:00 | 341.15 | 346.77 | 349.26 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 348.30 | 346.41 | 348.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 349.50 | 346.44 | 348.74 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 12:15:00 | 325.20 | 2024-11-25 14:15:00 | 347.45 | EXIT_EMA400 | -22.25 |
| BUY | 2025-07-14 09:15:00 | 345.00 | 2025-09-02 13:15:00 | 350.95 | EXIT_EMA400 | 5.95 |
| SELL | 2025-12-18 09:15:00 | 343.60 | 2026-01-05 11:15:00 | 349.50 | EXIT_EMA400 | -5.90 |
| SELL | 2025-12-30 10:15:00 | 341.15 | 2026-01-05 11:15:00 | 349.50 | EXIT_EMA400 | -8.35 |
