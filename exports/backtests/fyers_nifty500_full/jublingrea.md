# Jubilant Ingrevia Ltd. (JUBLINGREA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 706.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -89.11
- **Avg P&L per closed trade:** -11.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 10:15:00 | 690.30 | 753.48 | 753.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 686.95 | 752.82 | 753.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 734.00 | 726.03 | 737.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 11:15:00 | 712.50 | 727.34 | 736.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-07 10:15:00 | 696.40 | 669.86 | 696.17 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 687.00 | 681.07 | 681.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 692.50 | 681.26 | 681.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 720.55 | 682.40 | 681.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 720.55 | 682.40 | 681.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-02 11:15:00 | 728.25 | 687.47 | 684.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 695.00 | 696.97 | 690.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-11 10:15:00 | 684.85 | 696.82 | 690.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 701.00 | 744.97 | 745.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 690.10 | 723.49 | 731.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 686.00 | 680.94 | 701.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-07 13:15:00 | 648.30 | 684.58 | 695.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 708.95 | 680.80 | 691.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 707.60 | 698.41 | 698.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 719.00 | 698.70 | 698.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 683.30 | 698.44 | 698.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 680.55 | 698.26 | 698.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 678.55 | 693.66 | 695.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 09:15:00 | 698.70 | 692.78 | 695.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 710.60 | 697.10 | 697.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 722.70 | 697.49 | 697.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-31 10:15:00 | 708.50 | 699.43 | 698.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 706.30 | 710.07 | 704.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 09:15:00 | 700.00 | 709.97 | 704.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 669.80 | 699.89 | 699.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 667.45 | 699.57 | 699.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 674.05 | 668.58 | 681.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 655.50 | 668.29 | 681.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.50 | 661.35 | 675.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 11:15:00 | 663.95 | 661.43 | 675.87 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-11 10:15:00 | 675.85 | 661.65 | 675.55 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 717.05 | 623.93 | 623.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 10:15:00 | 738.45 | 625.07 | 624.25 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 11:15:00 | 712.50 | 2025-02-14 09:15:00 | 640.06 | TARGET | 72.44 |
| BUY | 2025-05-28 09:15:00 | 720.55 | 2025-06-11 10:15:00 | 684.85 | EXIT_EMA400 | -35.70 |
| BUY | 2025-06-02 11:15:00 | 728.25 | 2025-06-11 10:15:00 | 684.85 | EXIT_EMA400 | -43.40 |
| SELL | 2025-11-07 13:15:00 | 648.30 | 2025-11-17 09:15:00 | 708.95 | EXIT_EMA400 | -60.65 |
| SELL | 2025-12-18 09:15:00 | 678.55 | 2025-12-19 09:15:00 | 698.70 | EXIT_EMA400 | -20.15 |
| BUY | 2025-12-31 10:15:00 | 708.50 | 2026-01-05 09:15:00 | 739.10 | TARGET | 30.60 |
| SELL | 2026-02-04 09:15:00 | 655.50 | 2026-02-11 10:15:00 | 675.85 | EXIT_EMA400 | -20.35 |
| SELL | 2026-02-10 11:15:00 | 663.95 | 2026-02-11 10:15:00 | 675.85 | EXIT_EMA400 | -11.90 |
