# Travel Food Services Ltd. (TRAVELFOOD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-07-14 09:15:00 → 2026-04-30 15:15:00 (1374 bars)
- **Last close:** 1265.70
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
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 1 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 0 / 1
- **Target hits / EMA400 exits:** 0 / 1
- **Total realized P&L (per unit):** -48.30
- **Avg P&L per closed trade:** -48.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 1228.40 | 1291.07 | 1291.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 1224.10 | 1290.40 | 1290.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1136.00 | 1125.15 | 1171.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-09 11:15:00 | 1123.20 | 1125.13 | 1171.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 1171.50 | 1126.44 | 1171.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1287.40 | 1182.64 | 1182.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1293.10 | 1189.47 | 1185.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 12:15:00 | 1256.70 | 1259.64 | 1232.27 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2026-02-09 11:15:00 | 1123.20 | 2026-02-10 09:15:00 | 1171.50 | EXIT_EMA400 | -48.30 |
