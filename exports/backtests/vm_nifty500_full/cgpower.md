# CG Power and Industrial Solutions Ltd. (CGPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 813.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -79.99
- **Avg P&L per closed trade:** -8.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 14:15:00 | 380.15 | 415.56 | 415.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 10:15:00 | 379.95 | 414.52 | 415.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 14:15:00 | 395.90 | 392.99 | 401.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-22 09:15:00 | 387.70 | 392.78 | 399.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-22 11:15:00 | 411.90 | 392.91 | 399.84 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 13:15:00 | 438.20 | 406.32 | 406.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 14:15:00 | 438.75 | 406.64 | 406.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 454.65 | 454.69 | 441.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-17 12:15:00 | 463.95 | 454.89 | 442.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 446.95 | 455.25 | 444.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-24 09:15:00 | 434.55 | 454.96 | 444.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 12:15:00 | 424.75 | 441.81 | 441.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 13:15:00 | 424.15 | 441.63 | 441.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 439.50 | 439.49 | 440.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-27 13:15:00 | 437.50 | 439.47 | 440.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 437.00 | 439.44 | 440.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-28 12:15:00 | 433.30 | 439.31 | 440.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 435.00 | 438.86 | 440.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-29 14:15:00 | 443.00 | 438.91 | 440.24 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 10:15:00 | 470.25 | 441.56 | 441.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 09:15:00 | 474.60 | 450.22 | 446.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 491.50 | 493.79 | 476.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 14:15:00 | 495.20 | 493.66 | 477.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 556.95 | 605.82 | 564.61 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 11:15:00 | 716.30 | 734.67 | 734.67 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 761.00 | 734.58 | 734.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 762.85 | 735.08 | 734.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 736.05 | 736.92 | 735.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-29 13:15:00 | 742.40 | 736.98 | 735.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-29 14:15:00 | 730.50 | 736.92 | 735.75 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 705.35 | 743.73 | 743.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 695.90 | 743.26 | 743.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 608.25 | 603.54 | 637.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 14:15:00 | 595.90 | 607.22 | 634.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 633.10 | 607.63 | 630.68 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 665.90 | 626.22 | 626.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 678.50 | 627.54 | 626.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 15:15:00 | 673.50 | 674.13 | 658.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 679.50 | 673.88 | 659.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 667.45 | 677.19 | 666.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 14:15:00 | 665.80 | 676.90 | 666.42 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 689.45 | 728.29 | 728.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 14:15:00 | 688.60 | 727.51 | 728.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 609.35 | 606.62 | 638.76 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 711.70 | 656.58 | 656.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 719.20 | 657.75 | 657.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 684.20 | 692.87 | 679.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 698.30 | 692.81 | 679.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 682.70 | 694.35 | 681.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-20 14:15:00 | 680.45 | 694.21 | 681.95 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-22 09:15:00 | 387.70 | 2023-11-22 11:15:00 | 411.90 | EXIT_EMA400 | -24.20 |
| BUY | 2024-01-17 12:15:00 | 463.95 | 2024-01-24 09:15:00 | 434.55 | EXIT_EMA400 | -29.40 |
| SELL | 2024-02-27 13:15:00 | 437.50 | 2024-02-28 15:15:00 | 428.15 | TARGET | 9.35 |
| SELL | 2024-02-28 12:15:00 | 433.30 | 2024-02-29 14:15:00 | 443.00 | EXIT_EMA400 | -9.70 |
| BUY | 2024-04-15 14:15:00 | 495.20 | 2024-04-26 13:15:00 | 549.81 | TARGET | 54.61 |
| BUY | 2024-11-29 13:15:00 | 742.40 | 2024-11-29 14:15:00 | 730.50 | EXIT_EMA400 | -11.90 |
| SELL | 2025-03-10 14:15:00 | 595.90 | 2025-03-18 14:15:00 | 633.10 | EXIT_EMA400 | -37.20 |
| BUY | 2025-06-16 10:15:00 | 679.50 | 2025-07-02 14:15:00 | 665.80 | EXIT_EMA400 | -13.70 |
| BUY | 2026-03-16 14:15:00 | 698.30 | 2026-03-20 14:15:00 | 680.45 | EXIT_EMA400 | -17.85 |
