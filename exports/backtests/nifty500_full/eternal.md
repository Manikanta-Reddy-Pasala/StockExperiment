# Eternal Ltd. (ETERNAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-04-09 09:15:00 → 2026-04-30 15:30:00 (1812 bars)
- **Last close:** 247.03
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT3 | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 0 / 1
- **Target hits / EMA400 exits:** 0 / 1
- **Total realized P&L (per unit):** -16.65
- **Avg P&L per closed trade:** -16.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 306.85 | 317.57 | 317.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 303.30 | 317.32 | 317.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 291.05 | 288.78 | 296.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 10:15:00 | 279.95 | 288.47 | 296.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 293.20 | 288.28 | 296.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-14 11:15:00 | 296.60 | 288.73 | 295.93 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2026-01-12 10:15:00 | 279.95 | 2026-01-14 11:15:00 | 296.60 | EXIT_EMA400 | -16.65 |
