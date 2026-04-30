# Punjab National Bank (PNB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 109.59
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 3 / 0
- **Target hits / EMA400 exits:** 3 / 0
- **Total realized P&L (per unit):** 21.51
- **Avg P&L per closed trade:** 7.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 103.39 | 96.11 | 96.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 103.65 | 96.61 | 96.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 97.69 | 98.26 | 97.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-19 09:15:00 | 100.30 | 97.37 | 97.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 101.90 | 104.79 | 102.19 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 100.87 | 106.14 | 106.17 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 109.86 | 105.97 | 105.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 110.75 | 106.06 | 106.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 108.26 | 108.46 | 107.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 109.57 | 108.45 | 107.39 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 11:15:00 | 117.76 | 121.54 | 118.50 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 113.40 | 122.63 | 122.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 112.12 | 122.44 | 122.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 113.11 | 112.01 | 115.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 10:15:00 | 112.57 | 112.02 | 115.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 115.15 | 112.62 | 115.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-23 09:15:00 | 113.66 | 112.74 | 115.38 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-19 09:15:00 | 100.30 | 2025-06-04 09:15:00 | 110.11 | TARGET | 9.81 |
| BUY | 2025-09-29 09:15:00 | 109.57 | 2025-10-10 09:15:00 | 116.11 | TARGET | 6.54 |
| SELL | 2026-04-23 09:15:00 | 113.66 | 2026-04-30 10:15:00 | 108.50 | TARGET | 5.16 |
