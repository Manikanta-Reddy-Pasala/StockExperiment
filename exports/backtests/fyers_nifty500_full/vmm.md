# Vishal Mega Mart Ltd. (VMM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-18 09:15:00 → 2026-04-30 15:15:00 (2361 bars)
- **Last close:** 122.65
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
- **Total realized P&L (per unit):** 45.93
- **Avg P&L per closed trade:** 9.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 112.47 | 105.44 | 105.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 117.35 | 106.65 | 106.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 124.00 | 124.15 | 119.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 15:15:00 | 125.00 | 124.12 | 119.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 119.56 | 124.08 | 119.44 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-17 10:15:00 | 120.23 | 124.04 | 119.45 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 120.20 | 123.97 | 119.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-17 13:15:00 | 123.54 | 123.97 | 119.48 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 123.87 | 124.17 | 119.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-19 14:15:00 | 126.50 | 124.21 | 119.93 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 144.09 | 148.08 | 143.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 13:15:00 | 142.82 | 147.91 | 143.78 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 137.72 | 144.50 | 144.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 137.34 | 144.18 | 144.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 136.18 | 135.53 | 138.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 09:15:00 | 133.45 | 135.53 | 138.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 13:15:00 | 138.43 | 135.34 | 138.07 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-17 10:15:00 | 120.23 | 2025-06-17 13:15:00 | 122.58 | TARGET | 2.35 |
| BUY | 2025-06-17 13:15:00 | 123.54 | 2025-06-25 09:15:00 | 135.72 | TARGET | 12.18 |
| BUY | 2025-06-16 15:15:00 | 125.00 | 2025-07-22 09:15:00 | 141.67 | TARGET | 16.67 |
| BUY | 2025-06-19 14:15:00 | 126.50 | 2025-08-05 09:15:00 | 146.21 | TARGET | 19.71 |
| SELL | 2025-12-17 09:15:00 | 133.45 | 2025-12-19 13:15:00 | 138.43 | EXIT_EMA400 | -4.98 |
