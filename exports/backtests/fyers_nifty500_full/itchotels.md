# ITC Hotels Ltd. (ITCHOTELS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-01-29 09:15:00 → 2026-04-30 15:15:00 (2158 bars)
- **Last close:** 160.95
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
- **Total realized P&L (per unit):** 13.11
- **Avg P&L per closed trade:** 13.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 214.90 | 232.00 | 232.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 214.39 | 231.82 | 231.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 14:15:00 | 224.75 | 224.46 | 227.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-23 09:15:00 | 223.25 | 224.44 | 227.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 10:15:00 | 166.16 | 157.81 | 166.00 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-10-23 09:15:00 | 223.25 | 2025-11-07 09:15:00 | 210.14 | TARGET | 13.11 |
