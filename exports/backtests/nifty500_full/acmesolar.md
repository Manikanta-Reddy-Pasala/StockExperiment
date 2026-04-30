# ACME Solar Holdings Ltd. (ACMESOLAR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-13 09:15:00 → 2026-04-30 15:30:00 (2505 bars)
- **Last close:** 302.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 39.27
- **Avg P&L per closed trade:** 19.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 11:15:00 | 212.33 | 209.04 | 209.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 14:15:00 | 212.68 | 209.12 | 209.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 241.90 | 241.93 | 231.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 245.20 | 241.95 | 231.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 288.45 | 295.25 | 284.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-24 14:15:00 | 283.75 | 294.73 | 284.37 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 265.75 | 281.78 | 281.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 264.85 | 281.61 | 281.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 12:15:00 | 235.64 | 235.64 | 248.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 231.70 | 235.77 | 247.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 233.25 | 222.31 | 230.47 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 247.40 | 231.91 | 231.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 249.26 | 232.38 | 232.11 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-13 14:15:00 | 245.20 | 2025-07-10 13:15:00 | 286.02 | TARGET | 40.82 |
| SELL | 2025-12-26 13:15:00 | 231.70 | 2026-02-10 09:15:00 | 233.25 | EXIT_EMA400 | -1.55 |
