# Jio Financial Services Ltd. (JIOFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 247.18
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 21.92
- **Avg P&L per closed trade:** 4.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 350.60 | 339.08 | 339.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 355.00 | 340.08 | 339.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 344.70 | 345.23 | 342.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-03 14:15:00 | 346.95 | 345.13 | 342.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 346.95 | 345.13 | 342.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-04 09:15:00 | 342.40 | 345.11 | 342.69 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 330.00 | 341.26 | 341.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 328.55 | 340.80 | 341.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 323.45 | 322.14 | 328.21 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 340.90 | 331.30 | 331.28 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 314.25 | 331.17 | 331.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 13:15:00 | 312.45 | 330.81 | 331.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 230.05 | 228.67 | 247.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 10:15:00 | 224.75 | 229.14 | 244.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 238.21 | 227.11 | 238.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 15:15:00 | 239.20 | 227.23 | 238.38 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 255.40 | 244.65 | 244.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 260.40 | 245.41 | 245.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 283.30 | 283.58 | 272.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 288.30 | 283.62 | 272.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 313.30 | 322.32 | 313.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-28 10:15:00 | 315.80 | 322.26 | 313.19 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-28 13:15:00 | 311.70 | 322.00 | 313.20 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 294.50 | 311.01 | 311.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 291.90 | 305.13 | 306.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 302.30 | 300.48 | 303.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 13:15:00 | 298.60 | 300.45 | 303.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 301.55 | 298.92 | 301.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.98 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-03 14:15:00 | 346.95 | 2024-10-04 09:15:00 | 342.40 | EXIT_EMA400 | -4.55 |
| SELL | 2025-03-26 10:15:00 | 224.75 | 2025-04-15 15:15:00 | 239.20 | EXIT_EMA400 | -14.45 |
| BUY | 2025-06-20 09:15:00 | 288.30 | 2025-08-05 09:15:00 | 336.82 | TARGET | 48.52 |
| BUY | 2025-08-28 10:15:00 | 315.80 | 2025-08-28 13:15:00 | 311.70 | EXIT_EMA400 | -4.10 |
| SELL | 2025-12-24 13:15:00 | 298.60 | 2026-01-02 12:15:00 | 302.10 | EXIT_EMA400 | -3.50 |
