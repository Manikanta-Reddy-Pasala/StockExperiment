# Gillette India Ltd. (GILLETTE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7970.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -875.80
- **Avg P&L per closed trade:** -291.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 11:15:00 | 8677.20 | 9425.62 | 9426.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 8653.00 | 9351.43 | 9388.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 10:15:00 | 8763.40 | 8705.12 | 8988.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-19 15:15:00 | 8358.70 | 8699.28 | 8978.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 11:15:00 | 8302.00 | 8030.11 | 8261.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 8750.00 | 8271.88 | 8270.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 8885.50 | 8277.99 | 8273.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 10605.00 | 10658.42 | 10129.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-31 12:15:00 | 10989.00 | 10637.75 | 10173.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-08 14:15:00 | 10250.00 | 10631.95 | 10262.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 10000.00 | 10216.87 | 10217.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 9960.00 | 10214.31 | 10216.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 8834.00 | 8761.03 | 9152.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 11:15:00 | 8416.50 | 8738.28 | 9098.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 8223.00 | 8080.84 | 8332.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 8610.00 | 8090.15 | 8333.60 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-19 15:15:00 | 8358.70 | 2025-04-22 11:15:00 | 8302.00 | EXIT_EMA400 | 56.70 |
| BUY | 2025-07-31 12:15:00 | 10989.00 | 2025-08-08 14:15:00 | 10250.00 | EXIT_EMA400 | -739.00 |
| SELL | 2025-12-02 11:15:00 | 8416.50 | 2026-01-30 09:15:00 | 8610.00 | EXIT_EMA400 | -193.50 |
