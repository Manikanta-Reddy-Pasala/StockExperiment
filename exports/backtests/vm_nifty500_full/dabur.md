# Dabur India Ltd. (DABUR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 441.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 3 / 10
- **Target hits / EMA400 exits:** 3 / 10
- **Total realized P&L (per unit):** -58.84
- **Avg P&L per closed trade:** -4.53

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 12:15:00 | 553.70 | 564.91 | 564.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 10:15:00 | 549.40 | 564.34 | 564.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 565.90 | 562.37 | 563.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-09-20 09:15:00 | 555.20 | 563.77 | 564.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 555.20 | 563.77 | 564.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-09-25 14:15:00 | 554.25 | 562.68 | 563.43 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 540.70 | 535.18 | 542.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-20 09:15:00 | 542.65 | 535.26 | 542.40 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 549.90 | 543.20 | 543.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 11:15:00 | 565.00 | 543.95 | 543.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 546.50 | 547.37 | 545.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 10:15:00 | 553.35 | 547.42 | 545.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 550.25 | 548.93 | 546.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-17 13:15:00 | 551.95 | 548.96 | 546.64 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 539.75 | 548.88 | 546.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 532.25 | 544.69 | 544.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 525.85 | 537.96 | 540.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 529.50 | 529.33 | 534.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-04 12:15:00 | 514.10 | 529.27 | 533.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-02 14:15:00 | 522.85 | 513.32 | 521.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 557.00 | 527.32 | 527.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 559.30 | 528.21 | 527.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 626.35 | 627.48 | 608.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-21 12:15:00 | 631.80 | 623.67 | 610.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-25 09:15:00 | 626.20 | 650.19 | 635.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 567.50 | 626.63 | 626.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 563.75 | 599.19 | 610.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 518.25 | 514.84 | 532.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 505.45 | 515.22 | 532.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 523.30 | 514.34 | 529.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-09 11:15:00 | 521.00 | 514.40 | 529.89 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-17 10:15:00 | 530.05 | 515.40 | 527.56 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 512.80 | 485.02 | 484.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 513.10 | 485.30 | 485.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 513.00 | 513.79 | 504.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 09:15:00 | 515.45 | 513.81 | 504.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-11 09:15:00 | 502.25 | 513.52 | 504.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 490.35 | 516.10 | 516.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 489.10 | 514.73 | 515.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 505.60 | 505.10 | 509.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 09:15:00 | 491.90 | 505.34 | 508.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 503.80 | 503.82 | 507.76 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-04 10:15:00 | 508.55 | 503.87 | 507.76 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 526.65 | 510.77 | 510.71 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 503.05 | 511.79 | 511.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 502.20 | 511.55 | 511.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 505.35 | 500.55 | 504.99 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 522.30 | 508.08 | 508.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 13:15:00 | 523.85 | 508.37 | 508.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 11:15:00 | 509.70 | 510.49 | 509.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-21 12:15:00 | 513.20 | 510.27 | 509.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-28 09:15:00 | 508.20 | 512.47 | 510.63 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 505.10 | 509.22 | 509.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 499.75 | 509.13 | 509.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 510.70 | 508.94 | 509.09 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 514.70 | 509.26 | 509.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 516.25 | 509.33 | 509.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 15:15:00 | 510.10 | 511.12 | 510.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-16 10:15:00 | 515.00 | 511.15 | 510.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 515.00 | 511.15 | 510.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-17 09:15:00 | 518.75 | 511.39 | 510.41 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-02-18 09:15:00 | 506.45 | 511.63 | 510.57 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 485.70 | 510.10 | 510.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 481.10 | 508.88 | 509.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 450.25 | 443.67 | 463.38 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-09-20 09:15:00 | 555.20 | 2023-10-19 09:15:00 | 528.68 | TARGET | 26.52 |
| SELL | 2023-09-25 14:15:00 | 554.25 | 2023-10-19 09:15:00 | 526.70 | TARGET | 27.55 |
| BUY | 2024-01-11 10:15:00 | 553.35 | 2024-01-18 09:15:00 | 539.75 | EXIT_EMA400 | -13.60 |
| BUY | 2024-01-17 13:15:00 | 551.95 | 2024-01-18 09:15:00 | 539.75 | EXIT_EMA400 | -12.20 |
| SELL | 2024-04-04 12:15:00 | 514.10 | 2024-05-02 14:15:00 | 522.85 | EXIT_EMA400 | -8.75 |
| BUY | 2024-08-21 12:15:00 | 631.80 | 2024-09-25 09:15:00 | 626.20 | EXIT_EMA400 | -5.60 |
| SELL | 2025-01-06 09:15:00 | 505.45 | 2025-01-17 10:15:00 | 530.05 | EXIT_EMA400 | -24.60 |
| SELL | 2025-01-09 11:15:00 | 521.00 | 2025-01-17 10:15:00 | 530.05 | EXIT_EMA400 | -9.05 |
| BUY | 2025-08-07 09:15:00 | 515.45 | 2025-08-11 09:15:00 | 502.25 | EXIT_EMA400 | -13.20 |
| SELL | 2025-10-31 09:15:00 | 491.90 | 2025-11-04 10:15:00 | 508.55 | EXIT_EMA400 | -16.65 |
| BUY | 2026-01-21 12:15:00 | 513.20 | 2026-01-22 09:15:00 | 524.79 | TARGET | 11.59 |
| BUY | 2026-02-16 10:15:00 | 515.00 | 2026-02-18 09:15:00 | 506.45 | EXIT_EMA400 | -8.55 |
| BUY | 2026-02-17 09:15:00 | 518.75 | 2026-02-18 09:15:00 | 506.45 | EXIT_EMA400 | -12.30 |
