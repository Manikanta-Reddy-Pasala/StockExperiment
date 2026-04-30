# Federal Bank Ltd. (FEDERALBNK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 287.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 11.06
- **Avg P&L per closed trade:** 3.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 187.50 | 200.96 | 201.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 178.96 | 195.76 | 197.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 183.20 | 181.37 | 185.74 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 190.19 | 188.23 | 188.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 190.67 | 188.25 | 188.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 190.83 | 193.93 | 191.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 196.10 | 192.43 | 191.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 206.79 | 210.77 | 207.26 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 197.67 | 204.89 | 204.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 12:15:00 | 197.09 | 204.73 | 204.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 12:15:00 | 196.64 | 196.44 | 199.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-11 09:15:00 | 195.35 | 196.42 | 199.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 198.87 | 196.42 | 198.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 15:15:00 | 199.07 | 196.45 | 198.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 213.99 | 199.03 | 199.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 215.64 | 199.20 | 199.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 258.20 | 259.38 | 247.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 12:15:00 | 271.50 | 257.05 | 249.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 271.70 | 287.13 | 277.05 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 268.30 | 272.01 | 272.03 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 275.65 | 272.06 | 272.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 285.15 | 272.19 | 272.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 283.80 | 285.00 | 280.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 13:15:00 | 286.30 | 284.93 | 280.10 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-13 09:15:00 | 196.10 | 2025-06-03 11:15:00 | 210.68 | TARGET | 14.58 |
| SELL | 2025-09-11 09:15:00 | 195.35 | 2025-09-17 15:15:00 | 199.07 | EXIT_EMA400 | -3.72 |
| BUY | 2026-01-16 12:15:00 | 271.50 | 2026-03-09 09:15:00 | 271.70 | EXIT_EMA400 | 0.20 |
