# ITC Hotels Ltd. (ITCHOTELS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-01-29 09:15:00 → 2026-04-30 15:15:00 (2140 bars)
- **Last close:** 160.57
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
| ENTRY2 | 1 |
| EXIT | 0 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 1
- **Winners / losers:** 1 / 0
- **Target hits / EMA400 exits:** 1 / 0
- **Total realized P&L (per unit):** 16.05
- **Avg P&L per closed trade:** 16.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 214.90 | 232.00 | 232.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 214.45 | 231.82 | 231.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 224.32 | 224.24 | 227.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 12:15:00 | 222.00 | 224.21 | 227.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 166.16 | 157.61 | 166.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-22 09:15:00 | 163.20 | 158.05 | 166.14 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-10-24 12:15:00 | 222.00 | 2025-11-10 11:15:00 | 205.95 | TARGET | 16.05 |
