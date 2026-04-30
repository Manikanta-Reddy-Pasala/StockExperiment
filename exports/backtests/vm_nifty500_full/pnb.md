# Punjab National Bank (PNB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 109.36
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 3 / 1
- **Total realized P&L (per unit):** 16.86
- **Avg P&L per closed trade:** 4.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 121.64 | 125.31 | 125.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 10:15:00 | 121.02 | 125.23 | 125.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 126.91 | 121.15 | 122.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-31 09:15:00 | 123.81 | 121.82 | 122.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 123.81 | 121.82 | 122.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-07-31 10:15:00 | 124.73 | 121.85 | 123.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 103.39 | 96.11 | 96.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 103.65 | 96.61 | 96.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 97.72 | 98.26 | 97.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-19 09:15:00 | 100.29 | 97.37 | 97.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 101.91 | 104.78 | 102.20 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 100.87 | 106.14 | 106.17 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 109.89 | 105.97 | 105.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 110.75 | 106.06 | 106.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 108.26 | 108.46 | 107.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 109.55 | 108.45 | 107.39 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 11:15:00 | 117.76 | 121.54 | 118.49 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 113.40 | 122.65 | 122.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 112.11 | 122.45 | 122.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 113.10 | 112.02 | 115.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 10:15:00 | 112.54 | 112.02 | 115.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 115.00 | 112.59 | 115.42 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-22 13:15:00 | 114.90 | 112.63 | 115.42 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-07-31 09:15:00 | 123.81 | 2024-07-31 10:15:00 | 124.73 | EXIT_EMA400 | -0.92 |
| BUY | 2025-05-19 09:15:00 | 100.29 | 2025-06-04 09:15:00 | 110.04 | TARGET | 9.75 |
| BUY | 2025-09-29 09:15:00 | 109.55 | 2025-10-10 09:15:00 | 116.02 | TARGET | 6.47 |
| SELL | 2026-04-22 13:15:00 | 114.90 | 2026-04-23 09:15:00 | 113.34 | TARGET | 1.56 |
