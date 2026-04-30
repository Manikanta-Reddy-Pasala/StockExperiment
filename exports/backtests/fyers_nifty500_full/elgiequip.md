# Elgi Equipments Ltd. (ELGIEQUIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 547.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -59.54
- **Avg P&L per closed trade:** -5.41

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 595.70 | 669.24 | 669.46 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 726.90 | 667.39 | 667.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 730.40 | 687.28 | 678.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 698.60 | 705.10 | 691.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-24 11:15:00 | 709.45 | 705.14 | 691.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-25 11:15:00 | 692.00 | 704.93 | 692.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 670.00 | 684.45 | 684.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 665.70 | 683.00 | 683.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 656.40 | 650.23 | 664.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 13:15:00 | 623.10 | 648.51 | 660.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 15:15:00 | 650.00 | 617.71 | 640.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 535.00 | 480.58 | 480.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 536.35 | 481.13 | 480.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 508.65 | 510.42 | 499.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 524.80 | 510.53 | 499.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 543.10 | 552.93 | 537.15 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 531.75 | 552.72 | 537.12 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 495.65 | 527.03 | 527.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 489.10 | 526.36 | 526.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 496.35 | 492.38 | 505.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-29 14:15:00 | 475.20 | 493.76 | 503.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 500.00 | 491.67 | 500.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-08 09:15:00 | 487.40 | 491.88 | 500.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 478.05 | 482.30 | 492.67 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-23 10:15:00 | 474.80 | 482.23 | 492.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 490.70 | 481.67 | 491.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-27 10:15:00 | 487.75 | 481.73 | 491.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 490.50 | 482.12 | 491.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 494.75 | 482.24 | 491.58 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 508.10 | 494.41 | 494.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 509.95 | 494.99 | 494.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 496.70 | 496.76 | 495.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-05 10:15:00 | 500.00 | 496.72 | 495.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 09:15:00 | 492.35 | 496.80 | 495.71 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 484.35 | 494.74 | 494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 483.75 | 494.63 | 494.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 483.25 | 480.27 | 485.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-05 13:15:00 | 476.90 | 480.36 | 485.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 477.65 | 446.85 | 461.54 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 515.00 | 472.01 | 471.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 516.95 | 475.95 | 473.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 501.30 | 504.19 | 490.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 11:15:00 | 510.80 | 503.96 | 491.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 500.75 | 505.39 | 492.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-10 09:15:00 | 515.60 | 505.47 | 493.32 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-12 09:15:00 | 485.00 | 506.23 | 494.55 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 476.85 | 487.29 | 487.33 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 15:15:00 | 500.50 | 487.45 | 487.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 502.35 | 487.60 | 487.46 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-24 11:15:00 | 709.45 | 2024-09-25 11:15:00 | 692.00 | EXIT_EMA400 | -17.45 |
| SELL | 2024-11-11 13:15:00 | 623.10 | 2024-11-25 15:15:00 | 650.00 | EXIT_EMA400 | -26.90 |
| BUY | 2025-06-23 09:15:00 | 524.80 | 2025-07-24 09:15:00 | 601.24 | TARGET | 76.44 |
| SELL | 2025-09-29 14:15:00 | 475.20 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -19.55 |
| SELL | 2025-10-08 09:15:00 | 487.40 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -7.35 |
| SELL | 2025-10-23 10:15:00 | 474.80 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -19.95 |
| SELL | 2025-10-27 10:15:00 | 487.75 | 2025-10-28 09:15:00 | 494.75 | EXIT_EMA400 | -7.00 |
| BUY | 2025-12-05 10:15:00 | 500.00 | 2025-12-08 09:15:00 | 492.35 | EXIT_EMA400 | -7.65 |
| SELL | 2026-01-05 13:15:00 | 476.90 | 2026-01-09 09:15:00 | 450.63 | TARGET | 26.27 |
| BUY | 2026-03-05 11:15:00 | 510.80 | 2026-03-12 09:15:00 | 485.00 | EXIT_EMA400 | -25.80 |
| BUY | 2026-03-10 09:15:00 | 515.60 | 2026-03-12 09:15:00 | 485.00 | EXIT_EMA400 | -30.60 |
