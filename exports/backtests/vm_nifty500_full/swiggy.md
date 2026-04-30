# Swiggy Ltd. (SWIGGY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-13 09:15:00 → 2026-04-30 15:30:00 (2505 bars)
- **Last close:** 270.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 37.94
- **Avg P&L per closed trade:** 9.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 379.20 | 347.64 | 347.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 382.65 | 347.99 | 347.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 392.35 | 392.54 | 379.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 09:15:00 | 393.40 | 392.55 | 379.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 382.65 | 392.78 | 382.08 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-11 09:15:00 | 388.45 | 392.74 | 382.11 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 415.70 | 427.69 | 414.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 413.85 | 427.55 | 414.33 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 395.65 | 416.42 | 416.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 389.45 | 415.75 | 416.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 400.70 | 400.40 | 406.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-05 14:15:00 | 394.60 | 400.54 | 406.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 403.75 | 399.41 | 405.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-10 14:15:00 | 393.90 | 399.48 | 404.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-11 09:15:00 | 404.90 | 399.50 | 404.87 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-08-11 09:15:00 | 388.45 | 2025-08-18 09:15:00 | 407.46 | TARGET | 19.01 |
| BUY | 2025-08-04 09:15:00 | 393.40 | 2025-08-21 09:15:00 | 433.63 | TARGET | 40.23 |
| SELL | 2025-12-05 14:15:00 | 394.60 | 2025-12-11 09:15:00 | 404.90 | EXIT_EMA400 | -10.30 |
| SELL | 2025-12-10 14:15:00 | 393.90 | 2025-12-11 09:15:00 | 404.90 | EXIT_EMA400 | -11.00 |
