# Cemindia Projects Ltd. (CEMPRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-09-17 09:15:00 → 2026-04-30 15:15:00 (1048 bars)
- **Last close:** 815.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
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
- **Total realized P&L (per unit):** 164.64
- **Avg P&L per closed trade:** 164.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 790.50 | 816.79 | 816.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 775.35 | 813.15 | 814.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 696.50 | 696.11 | 737.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 681.55 | 696.05 | 736.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 604.25 | 560.87 | 603.65 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 12:15:00 | 815.25 | 625.23 | 625.00 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2026-02-05 09:15:00 | 681.55 | 2026-03-23 10:15:00 | 516.91 | TARGET | 164.64 |
