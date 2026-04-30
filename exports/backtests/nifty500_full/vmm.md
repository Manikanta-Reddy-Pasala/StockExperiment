# Vishal Mega Mart Ltd. (VMM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-18 09:15:00 → 2026-04-30 15:15:00 (2343 bars)
- **Last close:** 122.29
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| EXIT | 2 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 4 / 1
- **Total realized P&L (per unit):** 45.75
- **Avg P&L per closed trade:** 9.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 112.99 | 105.36 | 105.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 117.33 | 106.64 | 106.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 124.00 | 124.15 | 119.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 15:15:00 | 124.85 | 124.12 | 119.43 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 119.56 | 124.07 | 119.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-17 10:15:00 | 120.23 | 124.04 | 119.44 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 120.20 | 123.97 | 119.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-17 13:15:00 | 123.58 | 123.97 | 119.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 123.90 | 124.17 | 119.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-19 14:15:00 | 126.50 | 124.21 | 119.92 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 143.87 | 148.00 | 143.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 13:15:00 | 142.83 | 147.91 | 143.78 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 137.72 | 144.52 | 144.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 137.34 | 144.19 | 144.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 136.18 | 135.55 | 138.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 09:15:00 | 133.45 | 135.54 | 138.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 13:15:00 | 138.37 | 135.36 | 138.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-17 10:15:00 | 120.23 | 2025-06-17 13:15:00 | 122.60 | TARGET | 2.37 |
| BUY | 2025-06-17 13:15:00 | 123.58 | 2025-06-25 09:15:00 | 135.90 | TARGET | 12.32 |
| BUY | 2025-06-16 15:15:00 | 124.85 | 2025-07-22 09:15:00 | 141.10 | TARGET | 16.25 |
| BUY | 2025-06-19 14:15:00 | 126.50 | 2025-08-05 09:15:00 | 146.23 | TARGET | 19.73 |
| SELL | 2025-12-17 09:15:00 | 133.45 | 2025-12-19 13:15:00 | 138.37 | EXIT_EMA400 | -4.92 |
