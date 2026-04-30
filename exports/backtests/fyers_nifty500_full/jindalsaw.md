# Jindal Saw Ltd. (JINDALSAW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 224.49
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 103.60
- **Avg P&L per closed trade:** 20.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 325.85 | 334.87 | 334.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 322.00 | 334.74 | 334.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 316.65 | 314.95 | 322.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 11:15:00 | 307.90 | 319.82 | 322.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 308.80 | 319.31 | 322.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-18 10:15:00 | 306.85 | 319.18 | 322.43 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 261.25 | 247.26 | 260.74 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 189.53 | 174.46 | 174.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 177.11 | 175.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 180.15 | 182.13 | 178.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 184.83 | 181.35 | 178.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 184.83 | 181.35 | 178.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-25 10:15:00 | 187.62 | 181.41 | 179.03 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 181.70 | 182.25 | 179.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 11:15:00 | 177.72 | 182.18 | 179.69 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 165.30 | 177.69 | 177.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 188.06 | 177.77 | 177.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 190.20 | 178.01 | 177.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 183.78 | 184.87 | 181.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-02 14:15:00 | 192.35 | 185.38 | 182.67 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 11:15:00 | 307.90 | 2025-01-09 10:15:00 | 263.07 | TARGET | 44.83 |
| SELL | 2024-12-18 10:15:00 | 306.85 | 2025-01-09 13:15:00 | 260.10 | TARGET | 46.75 |
| BUY | 2026-02-25 09:15:00 | 184.83 | 2026-03-02 11:15:00 | 177.72 | EXIT_EMA400 | -7.11 |
| BUY | 2026-02-25 10:15:00 | 187.62 | 2026-03-02 11:15:00 | 177.72 | EXIT_EMA400 | -9.90 |
| BUY | 2026-04-02 14:15:00 | 192.35 | 2026-04-16 09:15:00 | 221.38 | TARGET | 29.03 |
