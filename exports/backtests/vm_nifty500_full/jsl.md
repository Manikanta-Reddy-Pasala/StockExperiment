# Jindal Stainless Ltd. (JSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 767.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -6.21
- **Avg P&L per closed trade:** -0.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 680.40 | 752.60 | 752.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 669.85 | 750.31 | 751.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 729.05 | 726.68 | 737.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-04 09:15:00 | 717.55 | 733.27 | 738.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 728.10 | 731.64 | 736.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-09 10:15:00 | 721.10 | 731.34 | 736.62 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-10 10:15:00 | 737.90 | 730.85 | 736.18 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 766.50 | 740.16 | 740.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 779.00 | 740.55 | 740.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 757.40 | 757.63 | 750.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 12:15:00 | 763.75 | 757.58 | 750.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-08 15:15:00 | 749.30 | 757.49 | 750.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 652.00 | 746.74 | 747.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 645.65 | 734.65 | 740.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 712.85 | 712.67 | 726.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 10:15:00 | 708.35 | 712.72 | 726.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 724.00 | 713.09 | 726.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-08 09:15:00 | 718.80 | 713.25 | 726.29 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 710.80 | 696.37 | 710.63 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 752.00 | 720.00 | 719.95 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 696.00 | 720.94 | 720.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 690.55 | 720.64 | 720.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 12:15:00 | 630.70 | 624.53 | 649.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-24 09:15:00 | 615.75 | 624.46 | 648.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 642.85 | 614.59 | 637.92 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 638.50 | 606.85 | 606.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 648.90 | 611.72 | 609.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 663.50 | 667.16 | 645.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 15:15:00 | 670.20 | 666.73 | 646.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 669.70 | 682.02 | 668.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 12:15:00 | 666.00 | 681.87 | 668.77 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 15:15:00 | 757.50 | 783.86 | 783.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 751.85 | 783.54 | 783.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 782.20 | 779.55 | 781.65 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 809.00 | 783.51 | 783.50 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 759.65 | 783.39 | 783.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 747.95 | 783.04 | 783.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 751.00 | 734.20 | 750.57 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 777.30 | 760.47 | 760.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 783.25 | 760.69 | 760.55 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-04 09:15:00 | 717.55 | 2024-09-10 10:15:00 | 737.90 | EXIT_EMA400 | -20.35 |
| SELL | 2024-09-09 10:15:00 | 721.10 | 2024-09-10 10:15:00 | 737.90 | EXIT_EMA400 | -16.80 |
| BUY | 2024-10-08 12:15:00 | 763.75 | 2024-10-08 15:15:00 | 749.30 | EXIT_EMA400 | -14.45 |
| SELL | 2024-11-08 09:15:00 | 718.80 | 2024-11-11 09:15:00 | 696.34 | TARGET | 22.46 |
| SELL | 2024-11-07 10:15:00 | 708.35 | 2024-11-21 10:15:00 | 654.12 | TARGET | 54.23 |
| SELL | 2025-02-24 09:15:00 | 615.75 | 2025-03-06 09:15:00 | 642.85 | EXIT_EMA400 | -27.10 |
| BUY | 2025-06-20 15:15:00 | 670.20 | 2025-07-24 12:15:00 | 666.00 | EXIT_EMA400 | -4.20 |
