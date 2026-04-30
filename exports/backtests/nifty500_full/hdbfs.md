# HDB Financial Services Ltd. (HDBFS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-07-02 09:15:00 → 2026-04-30 15:30:00 (1420 bars)
- **Last close:** 656.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 2 / 0
- **Target hits / EMA400 exits:** 2 / 0
- **Total realized P&L (per unit):** 75.01
- **Avg P&L per closed trade:** 37.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 763.10 | 753.53 | 753.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 765.00 | 753.64 | 753.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 753.00 | 756.89 | 755.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 12:15:00 | 762.15 | 756.28 | 755.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 752.05 | 758.53 | 756.50 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 15:15:00 | 712.00 | 754.50 | 754.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 705.50 | 750.59 | 752.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 725.00 | 724.94 | 734.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-20 15:15:00 | 716.00 | 724.85 | 734.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 690.60 | 633.31 | 663.73 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-01-12 12:15:00 | 762.15 | 2026-01-16 09:15:00 | 783.09 | TARGET | 20.94 |
| SELL | 2026-02-20 15:15:00 | 716.00 | 2026-03-09 09:15:00 | 661.93 | TARGET | 54.07 |
