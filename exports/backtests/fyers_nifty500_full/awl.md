# AWL Agri Business Ltd. (AWL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 195.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** -32.55
- **Avg P&L per closed trade:** -6.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 338.35 | 354.27 | 354.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 336.40 | 354.09 | 354.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 339.40 | 338.49 | 344.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 09:15:00 | 294.90 | 334.61 | 339.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 320.05 | 311.79 | 320.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-26 14:15:00 | 321.00 | 311.88 | 320.66 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 283.45 | 269.75 | 269.71 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 14:15:00 | 262.25 | 269.81 | 269.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 257.45 | 269.40 | 269.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 267.40 | 267.31 | 268.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-20 13:15:00 | 264.60 | 267.75 | 268.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-27 11:15:00 | 272.90 | 265.63 | 267.28 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 279.70 | 265.36 | 265.34 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 253.45 | 265.97 | 265.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 252.95 | 265.84 | 265.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 260.20 | 259.72 | 262.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 09:15:00 | 258.75 | 259.86 | 262.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 260.10 | 257.39 | 260.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-03 11:15:00 | 263.70 | 257.45 | 260.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 265.80 | 261.06 | 261.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 266.60 | 261.20 | 261.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-27 14:15:00 | 265.85 | 262.79 | 262.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 269.35 | 269.77 | 266.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-25 09:15:00 | 264.10 | 270.13 | 267.30 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 246.40 | 265.16 | 265.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 245.35 | 261.09 | 263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 11:15:00 | 196.28 | 196.17 | 210.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-12 09:15:00 | 176.60 | 195.43 | 209.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 09:15:00 | 196.18 | 184.24 | 193.11 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-21 09:15:00 | 294.90 | 2024-12-26 14:15:00 | 321.00 | EXIT_EMA400 | -26.10 |
| SELL | 2025-05-20 13:15:00 | 264.60 | 2025-05-27 11:15:00 | 272.90 | EXIT_EMA400 | -8.30 |
| SELL | 2025-08-22 09:15:00 | 258.75 | 2025-08-28 09:15:00 | 248.37 | TARGET | 10.38 |
| BUY | 2025-10-27 14:15:00 | 265.85 | 2025-11-04 14:15:00 | 276.90 | TARGET | 11.05 |
| SELL | 2026-03-12 09:15:00 | 176.60 | 2026-04-22 09:15:00 | 196.18 | EXIT_EMA400 | -19.58 |
