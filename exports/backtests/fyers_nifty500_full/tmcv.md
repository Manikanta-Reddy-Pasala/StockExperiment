# Tata Motors Ltd. (TMCV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-11-12 09:15:00 → 2026-04-30 15:15:00 (805 bars)
- **Last close:** 410.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
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

### Cycle 1 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 376.65 | 436.13 | 436.42 | EMA200 below EMA400 |

