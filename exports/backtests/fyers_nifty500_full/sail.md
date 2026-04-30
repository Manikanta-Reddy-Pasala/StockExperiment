# Steel Authority of India Ltd. (SAIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 184.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 3.93
- **Avg P&L per closed trade:** 0.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 116.92 | 110.97 | 110.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 117.75 | 111.20 | 111.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 103.48 | 111.97 | 111.50 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 102.00 | 110.97 | 111.01 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 114.03 | 111.03 | 111.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 116.20 | 111.14 | 111.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 112.87 | 113.18 | 112.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 116.11 | 113.21 | 112.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 111.82 | 113.31 | 112.41 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 121.35 | 126.97 | 127.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 121.17 | 126.92 | 126.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 124.39 | 124.04 | 125.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-02 15:15:00 | 122.90 | 124.00 | 125.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 125.00 | 124.01 | 125.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-03 10:15:00 | 126.50 | 124.04 | 125.23 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 131.75 | 126.17 | 126.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 131.80 | 126.23 | 126.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 15:15:00 | 130.60 | 130.76 | 128.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 133.22 | 130.79 | 128.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 131.25 | 132.02 | 130.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 11:15:00 | 131.62 | 132.01 | 130.14 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 11:15:00 | 129.64 | 131.97 | 130.18 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 129.83 | 133.45 | 133.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 127.59 | 132.96 | 133.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 147.34 | 133.22 | 133.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 13:15:00 | 149.20 | 136.21 | 134.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 143.94 | 146.51 | 142.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 14:15:00 | 147.95 | 146.46 | 142.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 146.98 | 156.77 | 151.66 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-02 09:15:00 | 116.11 | 2025-05-06 14:15:00 | 111.82 | EXIT_EMA400 | -4.29 |
| SELL | 2025-09-02 15:15:00 | 122.90 | 2025-09-03 10:15:00 | 126.50 | EXIT_EMA400 | -3.60 |
| BUY | 2025-09-29 09:15:00 | 133.22 | 2025-10-14 11:15:00 | 129.64 | EXIT_EMA400 | -3.58 |
| BUY | 2025-10-13 11:15:00 | 131.62 | 2025-10-14 11:15:00 | 129.64 | EXIT_EMA400 | -1.98 |
| BUY | 2026-02-02 14:15:00 | 147.95 | 2026-02-25 09:15:00 | 165.33 | TARGET | 17.38 |
