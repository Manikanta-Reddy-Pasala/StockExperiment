# LG Electronics India Ltd. (LGEINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-10-14 09:15:00 → 2026-04-30 15:30:00 (923 bars)
- **Last close:** 1593.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
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
- **Total realized P&L (per unit):** -27.60
- **Avg P&L per closed trade:** -27.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 1565.50 | 1525.07 | 1524.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 1577.40 | 1525.59 | 1525.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1523.60 | 1531.87 | 1528.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 09:15:00 | 1556.60 | 1531.98 | 1528.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-16 15:15:00 | 1529.00 | 1541.26 | 1534.26 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1455.80 | 1530.79 | 1530.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1447.20 | 1529.96 | 1530.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1494.80 | 1481.83 | 1502.46 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1592.10 | 1517.35 | 1517.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 1624.30 | 1518.42 | 1517.85 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-03-10 09:15:00 | 1556.60 | 2026-03-16 15:15:00 | 1529.00 | EXIT_EMA400 | -27.60 |
