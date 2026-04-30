# Emmvee Photovoltaic Power Ltd. (EMMVEE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-11-18 09:15:00 → 2026-04-30 15:15:00 (777 bars)
- **Last close:** 262.49
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT3 | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| EXIT | 0 |

## P&L

- **Trades closed:** 1
- **Trades open at end:** 0
- **Winners / losers:** 1 / 0
- **Target hits / EMA400 exits:** 1 / 0
- **Total realized P&L (per unit):** 13.31
- **Avg P&L per closed trade:** 13.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 217.93 | 203.51 | 203.45 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 14:15:00 | 185.95 | 203.88 | 203.91 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 227.99 | 203.46 | 203.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 238.56 | 204.29 | 203.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 10:15:00 | 211.07 | 212.86 | 208.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-02 12:15:00 | 213.39 | 212.86 | 208.95 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-04-02 12:15:00 | 213.39 | 2026-04-07 13:15:00 | 226.70 | TARGET | 13.31 |
