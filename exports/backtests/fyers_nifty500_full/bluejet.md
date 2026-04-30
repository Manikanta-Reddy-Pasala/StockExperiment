# Blue Jet Healthcare Ltd. (BLUEJET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 441.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT3 | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 1 / 0
- **Target hits / EMA400 exits:** 1 / 0
- **Total realized P&L (per unit):** 266.96
- **Avg P&L per closed trade:** 266.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 766.80 | 852.26 | 852.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 760.90 | 851.35 | 851.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 674.15 | 666.61 | 699.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-04 12:15:00 | 609.00 | 666.72 | 697.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 10:15:00 | 406.85 | 371.27 | 403.73 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-11-04 12:15:00 | 609.00 | 2026-03-23 14:15:00 | 342.04 | TARGET | 266.96 |
