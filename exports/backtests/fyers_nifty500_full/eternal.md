# Eternal Ltd. (ETERNAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 246.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 5 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -30.45
- **Avg P&L per closed trade:** -6.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 235.65 | 270.98 | 271.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 234.10 | 270.62 | 270.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 10:15:00 | 232.23 | 230.25 | 241.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-24 09:15:00 | 227.35 | 230.64 | 241.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 221.00 | 214.90 | 223.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-16 09:15:00 | 219.20 | 215.36 | 223.00 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 222.80 | 215.79 | 222.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 224.10 | 215.88 | 222.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 239.45 | 227.33 | 227.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 243.73 | 228.88 | 228.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 230.63 | 230.93 | 229.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 09:15:00 | 235.02 | 230.65 | 229.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 235.02 | 230.65 | 229.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-23 10:15:00 | 236.15 | 230.70 | 229.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 231.00 | 231.07 | 229.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-26 10:15:00 | 229.51 | 231.06 | 229.52 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 307.65 | 317.76 | 317.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 306.85 | 317.55 | 317.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 291.05 | 288.77 | 296.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 10:15:00 | 279.95 | 288.46 | 296.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 293.20 | 288.27 | 296.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-14 11:15:00 | 296.60 | 288.73 | 295.94 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-24 09:15:00 | 227.35 | 2025-04-17 10:15:00 | 224.10 | EXIT_EMA400 | 3.25 |
| SELL | 2025-04-16 09:15:00 | 219.20 | 2025-04-17 10:15:00 | 224.10 | EXIT_EMA400 | -4.90 |
| BUY | 2025-05-23 09:15:00 | 235.02 | 2025-05-26 10:15:00 | 229.51 | EXIT_EMA400 | -5.51 |
| BUY | 2025-05-23 10:15:00 | 236.15 | 2025-05-26 10:15:00 | 229.51 | EXIT_EMA400 | -6.64 |
| SELL | 2026-01-12 10:15:00 | 279.95 | 2026-01-14 11:15:00 | 296.60 | EXIT_EMA400 | -16.65 |
