# Gillette India Ltd. (GILLETTE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 7953.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 1311.08
- **Avg P&L per closed trade:** 262.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 6395.95 | 6525.95 | 6526.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 12:15:00 | 6354.20 | 6524.24 | 6525.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 6487.70 | 6468.83 | 6495.96 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 6761.90 | 6521.49 | 6520.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 10:15:00 | 6800.00 | 6524.26 | 6522.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 6848.05 | 6858.02 | 6734.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-04 12:15:00 | 7146.60 | 6860.89 | 6736.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 7146.60 | 6860.89 | 6736.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 09:15:00 | 7494.65 | 6874.55 | 6745.87 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 8464.55 | 8662.38 | 8380.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-04 11:15:00 | 8349.60 | 8645.69 | 8384.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 8680.90 | 9419.68 | 9421.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 15:15:00 | 8669.00 | 9398.14 | 9411.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 10:15:00 | 8758.65 | 8726.01 | 9005.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-19 15:15:00 | 8460.40 | 8720.16 | 8995.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 11:15:00 | 8302.00 | 8031.06 | 8265.66 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 8885.50 | 8278.28 | 8275.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 9375.00 | 8310.91 | 8292.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 10605.00 | 10658.22 | 10129.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-31 12:15:00 | 10989.00 | 10636.98 | 10173.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-08 14:15:00 | 10250.00 | 10631.54 | 10262.88 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 10012.00 | 10216.92 | 10217.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 9960.00 | 10214.36 | 10216.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 8834.00 | 8760.77 | 9152.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 11:15:00 | 8416.50 | 8737.99 | 9098.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 8223.00 | 8081.28 | 8332.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 8610.00 | 8090.54 | 8333.73 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-04 12:15:00 | 7146.60 | 2024-08-22 09:15:00 | 8376.83 | TARGET | 1230.23 |
| BUY | 2024-06-05 09:15:00 | 7494.65 | 2024-10-04 11:15:00 | 8349.60 | EXIT_EMA400 | 854.95 |
| SELL | 2025-02-19 15:15:00 | 8460.40 | 2025-04-22 11:15:00 | 8302.00 | EXIT_EMA400 | 158.40 |
| BUY | 2025-07-31 12:15:00 | 10989.00 | 2025-08-08 14:15:00 | 10250.00 | EXIT_EMA400 | -739.00 |
| SELL | 2025-12-02 11:15:00 | 8416.50 | 2026-01-30 09:15:00 | 8610.00 | EXIT_EMA400 | -193.50 |
