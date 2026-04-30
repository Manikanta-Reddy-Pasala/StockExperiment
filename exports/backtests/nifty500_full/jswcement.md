# JSW Cement Ltd. (JSWCEMENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-08-14 09:15:00 → 2026-04-30 15:30:00 (1203 bars)
- **Last close:** 122.47
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
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| EXIT | 0 |

## P&L

- **Trades closed:** 0
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / EMA400 exits:** 0 / 0
- **Total realized P&L (per unit):** 0.00
- **Avg P&L per closed trade:** 0.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 126.99 | 120.62 | 120.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 15:15:00 | 130.20 | 121.50 | 121.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 122.36 | 123.12 | 122.10 | EMA200 retest candle locked |

