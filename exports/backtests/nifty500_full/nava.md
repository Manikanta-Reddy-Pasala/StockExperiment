# Nava Ltd. (NAVA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 661.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -51.86
- **Avg P&L per closed trade:** -8.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 190.02 | 201.88 | 201.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 11:15:00 | 189.40 | 200.83 | 201.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 09:15:00 | 200.95 | 198.92 | 200.22 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 214.95 | 201.31 | 201.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 09:15:00 | 225.70 | 202.56 | 201.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-24 11:15:00 | 225.90 | 226.36 | 219.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-24 14:15:00 | 228.73 | 226.42 | 219.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 240.62 | 250.13 | 241.22 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 469.83 | 485.69 | 485.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 466.38 | 484.97 | 485.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 487.50 | 482.95 | 484.31 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 511.48 | 485.68 | 485.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 515.00 | 485.97 | 485.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 498.00 | 504.51 | 497.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 09:15:00 | 512.45 | 504.39 | 497.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 507.00 | 504.90 | 498.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 15:15:00 | 496.00 | 504.77 | 498.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 465.75 | 495.34 | 495.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 465.15 | 493.47 | 494.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 452.80 | 452.12 | 468.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 10:15:00 | 423.05 | 449.75 | 466.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 439.45 | 412.30 | 431.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 517.90 | 439.69 | 439.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 534.65 | 450.55 | 445.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 453.10 | 461.91 | 451.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 12:15:00 | 475.15 | 461.63 | 452.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-30 12:15:00 | 459.00 | 471.29 | 461.54 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 617.05 | 636.96 | 637.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 609.30 | 636.48 | 636.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 559.15 | 550.72 | 577.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 13:15:00 | 542.75 | 567.68 | 573.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 567.50 | 559.32 | 567.32 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 15:15:00 | 579.00 | 568.54 | 568.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 558.55 | 568.44 | 568.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 541.50 | 567.88 | 568.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 581.35 | 566.17 | 567.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-11 14:15:00 | 566.30 | 566.53 | 567.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 566.30 | 566.53 | 567.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-12 09:15:00 | 572.55 | 566.60 | 567.43 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 14:15:00 | 602.40 | 565.40 | 565.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 614.50 | 566.61 | 565.89 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-24 14:15:00 | 228.73 | 2024-02-02 09:15:00 | 256.87 | TARGET | 28.14 |
| BUY | 2024-12-16 09:15:00 | 512.45 | 2024-12-20 15:15:00 | 496.00 | EXIT_EMA400 | -16.45 |
| SELL | 2025-02-03 10:15:00 | 423.05 | 2025-03-06 14:15:00 | 439.45 | EXIT_EMA400 | -16.40 |
| BUY | 2025-04-08 12:15:00 | 475.15 | 2025-04-30 12:15:00 | 459.00 | EXIT_EMA400 | -16.15 |
| SELL | 2026-01-20 13:15:00 | 542.75 | 2026-01-30 09:15:00 | 567.50 | EXIT_EMA400 | -24.75 |
| SELL | 2026-03-11 14:15:00 | 566.30 | 2026-03-12 09:15:00 | 572.55 | EXIT_EMA400 | -6.25 |
