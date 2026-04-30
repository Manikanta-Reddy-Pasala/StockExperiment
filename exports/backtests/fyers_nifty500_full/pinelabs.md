# Pine Labs Ltd. (PINELABS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-11-14 09:15:00 → 2026-04-30 15:15:00 (791 bars)
- **Last close:** 195.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
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

_No CROSSOVER signals fired in window._
