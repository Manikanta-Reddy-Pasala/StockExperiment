# RBL Bank Ltd. (RBLBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 336.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 6.46
- **Avg P&L per closed trade:** 1.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 248.80 | 261.87 | 261.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 246.60 | 261.72 | 261.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 248.05 | 247.24 | 252.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-16 12:15:00 | 244.50 | 250.98 | 253.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-19 14:15:00 | 254.25 | 250.44 | 252.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 14:15:00 | 261.05 | 254.86 | 254.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 267.35 | 255.03 | 254.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 12:15:00 | 256.00 | 256.46 | 255.68 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 243.40 | 254.90 | 254.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 239.55 | 254.64 | 254.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 252.75 | 252.43 | 253.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 13:15:00 | 248.55 | 252.38 | 253.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 252.30 | 252.37 | 253.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-17 10:15:00 | 253.55 | 252.38 | 253.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 268.93 | 252.93 | 252.92 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 246.65 | 254.46 | 254.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 245.65 | 254.37 | 254.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 13:15:00 | 227.21 | 227.10 | 236.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-23 14:15:00 | 224.30 | 227.35 | 236.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 178.00 | 168.89 | 180.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 09:15:00 | 176.95 | 169.55 | 180.15 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 164.63 | 158.78 | 164.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 165.35 | 158.91 | 164.96 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 15:15:00 | 178.80 | 163.66 | 163.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 180.84 | 168.23 | 166.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 252.95 | 256.23 | 243.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 257.55 | 256.21 | 243.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 11:15:00 | 246.55 | 256.09 | 246.81 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 10:15:00 | 300.70 | 307.25 | 307.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 293.95 | 306.79 | 307.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 307.45 | 303.84 | 305.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 297.85 | 303.85 | 305.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 09:15:00 | 310.50 | 302.36 | 304.36 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 325.90 | 306.16 | 306.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 337.00 | 312.73 | 310.07 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-16 12:15:00 | 244.50 | 2024-04-19 14:15:00 | 254.25 | EXIT_EMA400 | -9.75 |
| SELL | 2024-05-16 13:15:00 | 248.55 | 2024-05-17 10:15:00 | 253.55 | EXIT_EMA400 | -5.00 |
| SELL | 2024-08-23 14:15:00 | 224.30 | 2024-10-21 09:15:00 | 189.04 | TARGET | 35.26 |
| SELL | 2024-12-12 09:15:00 | 176.95 | 2024-12-17 14:15:00 | 167.35 | TARGET | 9.60 |
| BUY | 2025-08-13 09:15:00 | 257.55 | 2025-08-28 11:15:00 | 246.55 | EXIT_EMA400 | -11.00 |
| SELL | 2026-03-27 09:15:00 | 297.85 | 2026-04-06 09:15:00 | 310.50 | EXIT_EMA400 | -12.65 |
