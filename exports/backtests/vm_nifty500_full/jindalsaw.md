# Jindal Saw Ltd. (JINDALSAW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 222.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 3 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / EMA400 exits:** 6 / 2
- **Total realized P&L (per unit):** 132.53
- **Avg P&L per closed trade:** 16.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 207.80 | 237.52 | 237.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 207.00 | 235.54 | 236.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 229.80 | 227.08 | 231.56 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 246.62 | 235.01 | 234.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 13:15:00 | 247.73 | 235.14 | 235.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 236.50 | 238.47 | 236.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 12:15:00 | 238.88 | 238.42 | 236.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 238.88 | 238.42 | 236.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 13:15:00 | 241.45 | 238.45 | 236.88 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 270.48 | 269.14 | 260.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 247.05 | 268.92 | 260.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 319.00 | 334.64 | 334.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 318.45 | 334.47 | 334.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 316.90 | 315.08 | 322.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 11:15:00 | 307.90 | 319.88 | 322.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 308.85 | 319.37 | 322.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-18 10:15:00 | 306.85 | 319.24 | 322.46 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 259.14 | 247.65 | 261.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-06 11:15:00 | 267.24 | 247.85 | 261.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 188.20 | 174.52 | 174.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 177.06 | 175.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 180.15 | 182.11 | 179.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 184.83 | 181.33 | 178.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 184.83 | 181.33 | 178.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-25 10:15:00 | 187.51 | 181.39 | 179.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 181.70 | 182.23 | 179.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 11:15:00 | 177.72 | 182.17 | 179.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 165.30 | 177.68 | 177.69 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 188.06 | 177.76 | 177.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 190.28 | 178.00 | 177.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 183.78 | 184.89 | 181.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-30 11:15:00 | 189.32 | 184.80 | 182.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 183.37 | 184.84 | 182.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-01 09:15:00 | 187.75 | 184.86 | 182.26 | Buy entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-19 12:15:00 | 238.88 | 2024-04-19 14:15:00 | 244.92 | TARGET | 6.04 |
| BUY | 2024-04-19 13:15:00 | 241.45 | 2024-04-24 09:15:00 | 255.15 | TARGET | 13.70 |
| SELL | 2024-12-17 11:15:00 | 307.90 | 2025-01-09 11:15:00 | 262.99 | TARGET | 44.91 |
| SELL | 2024-12-18 10:15:00 | 306.85 | 2025-01-09 14:15:00 | 260.02 | TARGET | 46.83 |
| BUY | 2026-02-25 09:15:00 | 184.83 | 2026-03-02 11:15:00 | 177.72 | EXIT_EMA400 | -7.11 |
| BUY | 2026-02-25 10:15:00 | 187.51 | 2026-03-02 11:15:00 | 177.72 | EXIT_EMA400 | -9.79 |
| BUY | 2026-04-01 09:15:00 | 187.75 | 2026-04-09 09:15:00 | 204.22 | TARGET | 16.47 |
| BUY | 2026-03-30 11:15:00 | 189.32 | 2026-04-15 09:15:00 | 210.79 | TARGET | 21.47 |
