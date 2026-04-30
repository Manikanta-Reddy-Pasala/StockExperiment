# ACME Solar Holdings Ltd. (ACMESOLAR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-13 09:15:00 → 2026-04-30 15:15:00 (2522 bars)
- **Last close:** 301.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 31.79
- **Avg P&L per closed trade:** 10.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 214.00 | 208.95 | 208.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 220.62 | 209.26 | 209.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 241.90 | 241.92 | 231.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 245.20 | 241.95 | 231.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 288.45 | 295.24 | 284.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-24 14:15:00 | 283.75 | 294.74 | 284.37 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 265.75 | 281.79 | 281.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 264.85 | 281.62 | 281.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 12:15:00 | 235.64 | 235.64 | 248.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 231.70 | 235.78 | 247.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 228.22 | 221.25 | 230.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 13:15:00 | 225.70 | 221.35 | 230.75 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 233.25 | 222.42 | 230.25 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 248.26 | 231.77 | 231.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 249.26 | 232.40 | 232.02 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-13 14:15:00 | 245.20 | 2025-07-10 13:15:00 | 286.09 | TARGET | 40.89 |
| SELL | 2025-12-26 13:15:00 | 231.70 | 2026-02-10 09:15:00 | 233.25 | EXIT_EMA400 | -1.55 |
| SELL | 2026-02-04 13:15:00 | 225.70 | 2026-02-10 09:15:00 | 233.25 | EXIT_EMA400 | -7.55 |
