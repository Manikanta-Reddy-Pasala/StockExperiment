# LG Electronics India Ltd. (LGEINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-10-14 09:15:00 → 2026-04-30 15:15:00 (933 bars)
- **Last close:** 1586.00
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
- **Total realized P&L (per unit):** -26.80
- **Avg P&L per closed trade:** -26.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 10:15:00 | 1553.90 | 1522.60 | 1522.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 1577.40 | 1525.11 | 1523.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1523.60 | 1531.45 | 1527.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 09:15:00 | 1556.60 | 1531.58 | 1527.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-16 15:15:00 | 1529.80 | 1540.96 | 1533.20 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 1427.00 | 1528.50 | 1528.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1402.00 | 1523.82 | 1526.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1495.50 | 1481.57 | 1501.76 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 1544.00 | 1516.43 | 1516.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 1580.00 | 1518.07 | 1517.21 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-03-10 09:15:00 | 1556.60 | 2026-03-16 15:15:00 | 1529.80 | EXIT_EMA400 | -26.80 |
