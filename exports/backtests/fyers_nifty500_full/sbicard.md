# SBI Cards and Payment Services Ltd. (SBICARD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 645.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 69.60
- **Avg P&L per closed trade:** 13.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 702.65 | 718.37 | 718.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 13:15:00 | 702.00 | 718.21 | 718.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 712.55 | 712.32 | 715.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-21 12:15:00 | 709.95 | 712.29 | 715.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-22 11:15:00 | 715.35 | 712.25 | 714.92 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 725.90 | 717.05 | 717.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 731.50 | 717.41 | 717.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 10:15:00 | 769.00 | 770.33 | 753.13 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 707.00 | 745.00 | 745.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 696.95 | 742.26 | 743.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 702.00 | 700.84 | 715.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 685.65 | 711.60 | 715.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 703.05 | 698.62 | 707.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-03 11:15:00 | 708.25 | 698.76 | 707.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 747.90 | 713.29 | 713.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 751.80 | 713.67 | 713.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 841.50 | 844.22 | 819.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 10:15:00 | 850.70 | 842.13 | 821.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 11:15:00 | 924.30 | 954.31 | 925.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 844.10 | 914.46 | 914.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 840.10 | 913.72 | 914.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 819.65 | 818.61 | 844.11 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 890.95 | 857.95 | 857.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 899.00 | 858.36 | 858.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 898.00 | 899.93 | 883.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 14:15:00 | 906.50 | 899.80 | 884.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-30 14:15:00 | 885.35 | 900.44 | 885.68 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 870.25 | 879.77 | 879.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 879.48 | 879.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 09:15:00 | 868.25 | 877.04 | 878.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 11:15:00 | 878.70 | 869.84 | 874.14 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-21 12:15:00 | 709.95 | 2024-08-22 11:15:00 | 715.35 | EXIT_EMA400 | -5.40 |
| SELL | 2024-12-20 14:15:00 | 685.65 | 2025-01-03 11:15:00 | 708.25 | EXIT_EMA400 | -22.60 |
| BUY | 2025-04-11 10:15:00 | 850.70 | 2025-06-04 09:15:00 | 939.30 | TARGET | 88.60 |
| BUY | 2025-10-28 14:15:00 | 906.50 | 2025-10-30 14:15:00 | 885.35 | EXIT_EMA400 | -21.15 |
| SELL | 2025-12-15 09:15:00 | 868.25 | 2025-12-17 09:15:00 | 838.09 | TARGET | 30.16 |
