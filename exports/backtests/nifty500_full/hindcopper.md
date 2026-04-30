# Hindustan Copper Ltd. (HINDCOPPER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 534.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 33.35
- **Avg P&L per closed trade:** 5.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 12:15:00 | 330.35 | 335.47 | 335.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 329.30 | 335.41 | 335.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 327.20 | 325.63 | 329.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-01 10:15:00 | 324.40 | 325.62 | 329.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 320.80 | 313.54 | 321.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-19 10:15:00 | 321.80 | 313.62 | 321.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 340.55 | 321.63 | 321.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 343.70 | 322.22 | 321.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 320.30 | 328.01 | 325.15 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 317.25 | 322.89 | 322.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 313.00 | 322.76 | 322.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 324.50 | 322.57 | 322.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-18 15:15:00 | 320.35 | 322.55 | 322.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 320.35 | 322.55 | 322.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-21 09:15:00 | 318.00 | 322.50 | 322.70 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 292.15 | 284.16 | 294.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-10 12:15:00 | 289.60 | 284.61 | 293.95 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-11 09:15:00 | 294.05 | 284.91 | 293.91 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 245.65 | 224.24 | 224.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 253.69 | 224.73 | 224.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 243.69 | 245.42 | 237.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-19 13:15:00 | 245.82 | 245.43 | 237.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 15:15:00 | 256.20 | 265.49 | 256.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 239.80 | 251.56 | 251.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 239.09 | 251.01 | 251.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 243.80 | 242.88 | 246.49 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 283.30 | 248.60 | 248.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 289.62 | 249.38 | 248.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 325.45 | 329.04 | 307.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 337.55 | 328.91 | 308.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-21 14:15:00 | 314.25 | 331.17 | 316.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 09:15:00 | 482.10 | 514.79 | 514.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 471.30 | 513.14 | 513.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 526.95 | 511.15 | 512.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 519.95 | 514.23 | 514.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 519.95 | 514.23 | 514.29 | EMA400 retest candle locked |

### Cycle 8 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 528.65 | 514.37 | 514.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 531.45 | 514.54 | 514.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 534.40 | 536.47 | 527.74 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-01 10:15:00 | 324.40 | 2024-08-02 09:15:00 | 308.75 | TARGET | 15.65 |
| SELL | 2024-10-18 15:15:00 | 320.35 | 2024-10-22 09:15:00 | 313.23 | TARGET | 7.12 |
| SELL | 2024-10-21 09:15:00 | 318.00 | 2024-10-22 13:15:00 | 303.90 | TARGET | 14.10 |
| SELL | 2024-12-10 12:15:00 | 289.60 | 2024-12-11 09:15:00 | 294.05 | EXIT_EMA400 | -4.45 |
| BUY | 2025-06-19 13:15:00 | 245.82 | 2025-06-26 14:15:00 | 270.04 | TARGET | 24.22 |
| BUY | 2025-11-10 09:15:00 | 337.55 | 2025-11-21 14:15:00 | 314.25 | EXIT_EMA400 | -23.30 |
