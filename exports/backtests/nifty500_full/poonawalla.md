# Poonawalla Fincorp Ltd. (POONAWALLA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 417.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -40.04
- **Avg P&L per closed trade:** -6.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 15:15:00 | 379.00 | 388.49 | 388.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 375.75 | 388.02 | 388.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 379.55 | 367.20 | 375.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-20 11:15:00 | 364.35 | 374.06 | 377.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-29 13:15:00 | 375.40 | 370.43 | 374.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 415.40 | 378.50 | 378.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 14:15:00 | 420.75 | 384.55 | 381.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 470.50 | 471.51 | 450.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 13:15:00 | 482.65 | 471.93 | 451.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 458.00 | 474.95 | 455.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-28 14:15:00 | 455.00 | 472.04 | 458.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 458.25 | 471.98 | 472.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 456.10 | 471.82 | 471.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 10:15:00 | 462.00 | 458.79 | 464.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-10 10:15:00 | 447.95 | 459.01 | 464.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 399.75 | 380.24 | 400.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-21 09:15:00 | 402.20 | 380.46 | 400.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 348.45 | 314.12 | 313.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 358.25 | 317.82 | 315.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 446.00 | 446.62 | 425.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 450.10 | 437.69 | 428.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 14:15:00 | 427.85 | 448.85 | 437.51 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 461.90 | 474.82 | 474.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 460.20 | 474.68 | 474.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 463.95 | 460.47 | 466.46 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 12:15:00 | 472.40 | 470.23 | 470.23 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 467.95 | 470.21 | 470.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 463.00 | 470.14 | 470.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 13:15:00 | 468.85 | 467.10 | 468.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 12:15:00 | 463.95 | 467.16 | 468.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-19 09:15:00 | 477.75 | 467.20 | 468.52 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 463.75 | 457.37 | 457.37 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 454.00 | 457.34 | 457.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 442.50 | 457.04 | 457.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 425.15 | 409.61 | 425.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 10:15:00 | 410.35 | 410.24 | 425.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 424.80 | 409.16 | 421.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-20 11:15:00 | 364.35 | 2023-11-29 13:15:00 | 375.40 | EXIT_EMA400 | -11.05 |
| BUY | 2024-02-13 13:15:00 | 482.65 | 2024-02-28 14:15:00 | 455.00 | EXIT_EMA400 | -27.65 |
| SELL | 2024-06-10 10:15:00 | 447.95 | 2024-07-23 11:15:00 | 398.79 | TARGET | 49.16 |
| BUY | 2025-08-13 09:15:00 | 450.10 | 2025-08-28 14:15:00 | 427.85 | EXIT_EMA400 | -22.25 |
| SELL | 2026-01-16 12:15:00 | 463.95 | 2026-01-19 09:15:00 | 477.75 | EXIT_EMA400 | -13.80 |
| SELL | 2026-04-09 10:15:00 | 410.35 | 2026-04-21 09:15:00 | 424.80 | EXIT_EMA400 | -14.45 |
