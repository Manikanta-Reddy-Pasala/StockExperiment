# Petronet LNG Ltd. (PETRONET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 276.76
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 42.17
- **Avg P&L per closed trade:** 5.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 12:15:00 | 247.00 | 224.85 | 224.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 249.55 | 225.73 | 225.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 12:15:00 | 234.60 | 235.34 | 231.69 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 13:15:00 | 218.15 | 229.84 | 229.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 13:15:00 | 214.25 | 228.55 | 229.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 205.20 | 204.34 | 212.17 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 228.60 | 214.16 | 214.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 13:15:00 | 229.50 | 214.45 | 214.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 274.05 | 274.22 | 260.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-03 09:15:00 | 279.15 | 268.59 | 261.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 293.50 | 300.41 | 291.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-31 13:15:00 | 297.10 | 300.32 | 291.94 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 289.10 | 300.93 | 292.70 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 331.15 | 344.46 | 344.52 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 13:15:00 | 363.25 | 344.56 | 344.53 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 325.95 | 346.03 | 346.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 322.35 | 340.49 | 342.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 332.60 | 331.95 | 337.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 13:15:00 | 330.70 | 332.07 | 337.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 333.10 | 331.22 | 336.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 339.05 | 331.29 | 336.28 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 346.55 | 338.06 | 338.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 348.25 | 338.16 | 338.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 326.80 | 338.32 | 338.16 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 323.10 | 337.88 | 337.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 321.85 | 334.88 | 336.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 12:15:00 | 329.95 | 329.85 | 332.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 324.10 | 329.80 | 332.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 13:15:00 | 304.05 | 293.85 | 303.55 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 310.45 | 303.04 | 303.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 311.95 | 304.17 | 303.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 311.20 | 312.35 | 308.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 15:15:00 | 314.35 | 312.38 | 308.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-30 10:15:00 | 308.80 | 312.33 | 308.94 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 295.30 | 307.17 | 307.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 292.00 | 306.78 | 307.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 305.05 | 303.71 | 305.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-30 12:15:00 | 300.65 | 303.61 | 305.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 302.55 | 302.76 | 304.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-07 10:15:00 | 307.65 | 302.81 | 304.48 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 288.35 | 277.23 | 277.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 11:15:00 | 293.30 | 278.24 | 277.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 282.00 | 282.28 | 280.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-28 09:15:00 | 288.10 | 280.47 | 279.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 288.10 | 280.47 | 279.58 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-28 10:15:00 | 291.65 | 280.58 | 279.64 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 277.35 | 299.68 | 292.55 | Close below EMA400 |

### Cycle 12 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 238.30 | 288.44 | 288.68 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-31 13:15:00 | 297.10 | 2024-06-03 14:15:00 | 312.59 | TARGET | 15.49 |
| BUY | 2024-04-03 09:15:00 | 279.15 | 2024-06-04 10:15:00 | 289.10 | EXIT_EMA400 | 9.95 |
| SELL | 2024-11-26 13:15:00 | 330.70 | 2024-12-02 09:15:00 | 339.05 | EXIT_EMA400 | -8.35 |
| SELL | 2025-01-24 09:15:00 | 324.10 | 2025-01-28 13:15:00 | 297.73 | TARGET | 26.37 |
| BUY | 2025-05-29 15:15:00 | 314.35 | 2025-05-30 10:15:00 | 308.80 | EXIT_EMA400 | -5.55 |
| SELL | 2025-06-30 12:15:00 | 300.65 | 2025-07-07 10:15:00 | 307.65 | EXIT_EMA400 | -7.00 |
| BUY | 2026-01-28 09:15:00 | 288.10 | 2026-02-25 10:15:00 | 313.66 | TARGET | 25.56 |
| BUY | 2026-01-28 10:15:00 | 291.65 | 2026-03-04 09:15:00 | 277.35 | EXIT_EMA400 | -14.30 |
