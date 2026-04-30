# Tata Investment Corporation Ltd. (TATAINVEST.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 719.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -202.49
- **Avg P&L per closed trade:** -40.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 633.96 | 654.44 | 654.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 11:15:00 | 631.52 | 654.21 | 654.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 630.51 | 622.40 | 633.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-23 10:15:00 | 617.80 | 623.00 | 633.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-27 09:15:00 | 664.60 | 622.65 | 632.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 715.13 | 641.59 | 641.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 753.34 | 642.70 | 641.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 678.84 | 680.98 | 666.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-23 10:15:00 | 709.92 | 679.61 | 667.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-03 09:15:00 | 670.98 | 683.06 | 671.77 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 649.13 | 673.48 | 673.60 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 689.53 | 673.41 | 673.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 696.00 | 677.49 | 675.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 679.20 | 679.90 | 676.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 14:15:00 | 689.00 | 680.10 | 677.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-17 12:15:00 | 677.30 | 680.71 | 677.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 645.24 | 676.43 | 676.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 638.63 | 675.42 | 675.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 594.75 | 589.53 | 618.55 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 09:15:00 | 623.30 | 619.46 | 619.45 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 11:15:00 | 611.45 | 619.48 | 619.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 14:15:00 | 610.35 | 619.25 | 619.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 607.50 | 607.14 | 612.40 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 635.85 | 615.59 | 615.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 646.70 | 616.09 | 615.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 654.50 | 657.24 | 641.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 14:15:00 | 661.65 | 656.42 | 642.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 658.80 | 667.26 | 655.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 15:15:00 | 654.10 | 666.59 | 655.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 758.30 | 788.39 | 788.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 748.00 | 787.68 | 788.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 654.55 | 651.35 | 685.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 626.60 | 652.58 | 681.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-19 09:15:00 | 724.10 | 649.24 | 676.12 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 728.00 | 651.62 | 651.56 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-23 10:15:00 | 617.80 | 2024-08-27 09:15:00 | 664.60 | EXIT_EMA400 | -46.80 |
| BUY | 2024-09-23 10:15:00 | 709.92 | 2024-10-03 09:15:00 | 670.98 | EXIT_EMA400 | -38.94 |
| BUY | 2024-12-13 14:15:00 | 689.00 | 2024-12-17 12:15:00 | 677.30 | EXIT_EMA400 | -11.70 |
| BUY | 2025-06-20 14:15:00 | 661.65 | 2025-07-14 15:15:00 | 654.10 | EXIT_EMA400 | -7.55 |
| SELL | 2026-02-13 09:15:00 | 626.60 | 2026-02-19 09:15:00 | 724.10 | EXIT_EMA400 | -97.50 |
