# Elgi Equipments Ltd. (ELGIEQUIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 553.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 6 |
| EXIT | 8 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / EMA400 exits:** 4 / 10
- **Total realized P&L (per unit):** 100.00
- **Avg P&L per closed trade:** 7.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 526.45 | 504.57 | 504.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 532.35 | 512.89 | 509.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 522.50 | 523.62 | 516.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 550.15 | 523.29 | 517.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 532.40 | 533.76 | 527.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-18 11:15:00 | 548.00 | 533.91 | 527.20 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 615.60 | 642.83 | 610.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 10:15:00 | 632.80 | 641.29 | 611.32 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-20 09:15:00 | 611.05 | 638.76 | 613.78 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 14:15:00 | 589.05 | 623.27 | 623.29 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 13:15:00 | 688.95 | 623.14 | 622.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 10:15:00 | 693.05 | 625.63 | 624.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 697.55 | 698.61 | 673.51 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 598.30 | 670.12 | 670.32 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 721.35 | 667.92 | 667.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 730.40 | 687.26 | 679.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 698.60 | 705.13 | 691.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-24 10:15:00 | 705.05 | 705.13 | 691.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-25 11:15:00 | 692.00 | 704.96 | 692.18 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 670.20 | 684.61 | 684.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 659.95 | 682.08 | 683.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 657.00 | 650.17 | 664.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 13:15:00 | 623.10 | 648.32 | 661.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 15:15:00 | 640.75 | 617.48 | 640.24 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 534.65 | 480.62 | 480.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 536.35 | 481.17 | 480.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 508.25 | 510.41 | 499.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 525.20 | 510.52 | 499.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 543.10 | 552.95 | 537.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 531.75 | 552.74 | 537.15 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 495.65 | 527.08 | 527.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 489.10 | 526.41 | 526.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 496.50 | 492.36 | 505.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-29 14:15:00 | 475.20 | 493.71 | 503.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 500.00 | 491.64 | 500.59 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-08 09:15:00 | 487.40 | 491.88 | 500.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 478.05 | 482.29 | 492.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-23 10:15:00 | 474.80 | 482.22 | 492.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 490.70 | 481.65 | 491.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-27 10:15:00 | 487.75 | 481.71 | 491.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 490.30 | 482.09 | 491.55 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 494.75 | 482.22 | 491.57 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 508.10 | 494.42 | 494.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 509.95 | 495.01 | 494.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 496.70 | 496.77 | 495.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-05 10:15:00 | 500.00 | 496.74 | 495.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 09:15:00 | 492.35 | 496.81 | 495.72 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 484.35 | 494.74 | 494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 483.75 | 494.63 | 494.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 483.25 | 480.29 | 485.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-05 13:15:00 | 476.90 | 480.38 | 485.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 477.65 | 447.33 | 462.30 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 513.40 | 472.68 | 472.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 515.75 | 474.12 | 473.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 501.30 | 504.29 | 491.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 11:15:00 | 510.80 | 504.05 | 491.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 500.75 | 505.45 | 493.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-09 12:15:00 | 509.55 | 505.40 | 493.33 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-12 09:15:00 | 485.00 | 506.28 | 494.84 | Close below EMA400 |

### Cycle 12 — SELL (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 14:15:00 | 480.00 | 487.65 | 487.68 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 09:15:00 | 502.35 | 487.64 | 487.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 517.45 | 488.68 | 488.17 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-18 11:15:00 | 548.00 | 2024-01-30 09:15:00 | 610.40 | TARGET | 62.40 |
| BUY | 2023-12-22 09:15:00 | 550.15 | 2024-02-07 09:15:00 | 648.29 | TARGET | 98.14 |
| BUY | 2024-03-14 10:15:00 | 632.80 | 2024-03-20 09:15:00 | 611.05 | EXIT_EMA400 | -21.75 |
| BUY | 2024-09-24 10:15:00 | 705.05 | 2024-09-25 11:15:00 | 692.00 | EXIT_EMA400 | -13.05 |
| SELL | 2024-11-11 13:15:00 | 623.10 | 2024-11-25 15:15:00 | 640.75 | EXIT_EMA400 | -17.65 |
| BUY | 2025-06-23 09:15:00 | 525.20 | 2025-07-24 09:15:00 | 602.66 | TARGET | 77.46 |
| SELL | 2025-09-29 14:15:00 | 475.20 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -19.55 |
| SELL | 2025-10-08 09:15:00 | 487.40 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -7.35 |
| SELL | 2025-10-23 10:15:00 | 474.80 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -19.95 |
| SELL | 2025-10-27 10:15:00 | 487.75 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -7.00 |
| BUY | 2025-12-05 10:15:00 | 500.00 | 2025-12-08 09:15:00 | 492.35 | EXIT_EMA400 | -7.65 |
| SELL | 2026-01-05 13:15:00 | 476.90 | 2026-01-09 09:15:00 | 450.60 | TARGET | 26.30 |
| BUY | 2026-03-05 11:15:00 | 510.80 | 2026-03-12 09:15:00 | 485.00 | EXIT_EMA400 | -25.80 |
| BUY | 2026-03-09 12:15:00 | 509.55 | 2026-03-12 09:15:00 | 485.00 | EXIT_EMA400 | -24.55 |
