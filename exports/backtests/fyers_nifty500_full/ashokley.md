# Ashok Leyland Ltd. (ASHOKLEY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 162.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -3.59
- **Avg P&L per closed trade:** -1.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 13:15:00 | 110.39 | 120.18 | 120.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 09:15:00 | 108.82 | 119.31 | 119.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 111.00 | 110.03 | 113.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 107.63 | 110.24 | 113.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 112.28 | 110.10 | 112.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-19 14:15:00 | 110.45 | 110.11 | 112.84 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 114.70 | 110.20 | 112.67 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 115.31 | 106.40 | 106.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 115.60 | 109.60 | 108.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 116.44 | 117.38 | 114.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 116.99 | 117.38 | 114.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 119.55 | 122.97 | 120.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 10:15:00 | 172.24 | 185.63 | 185.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 171.07 | 185.08 | 185.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 175.71 | 173.61 | 178.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 13:15:00 | 170.15 | 174.62 | 177.93 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 09:15:00 | 107.63 | 2024-11-25 09:15:00 | 114.70 | EXIT_EMA400 | -7.07 |
| SELL | 2024-11-19 14:15:00 | 110.45 | 2024-11-25 09:15:00 | 114.70 | EXIT_EMA400 | -4.25 |
| BUY | 2025-06-13 10:15:00 | 116.99 | 2025-06-27 09:15:00 | 124.72 | TARGET | 7.73 |
