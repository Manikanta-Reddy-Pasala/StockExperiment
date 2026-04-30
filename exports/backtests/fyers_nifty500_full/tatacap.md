# Tata Capital Ltd. (TATACAP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-10-13 09:15:00 → 2026-04-30 15:15:00 (940 bars)
- **Last close:** 334.00
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
- **Total realized P&L (per unit):** -6.35
- **Avg P&L per closed trade:** -6.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 09:15:00 | 322.25 | 340.71 | 340.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 321.40 | 340.52 | 340.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 327.20 | 325.83 | 331.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-25 11:15:00 | 321.80 | 325.74 | 331.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 326.50 | 320.47 | 327.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 10:15:00 | 328.15 | 320.66 | 327.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2026-03-25 11:15:00 | 321.80 | 2026-04-09 10:15:00 | 328.15 | EXIT_EMA400 | -6.35 |
