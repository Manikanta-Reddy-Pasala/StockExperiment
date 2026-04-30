# Asahi India Glass Ltd. (ASAHIINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 836.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 45.44
- **Avg P&L per closed trade:** 6.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 13:15:00 | 545.90 | 567.89 | 568.00 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 578.10 | 566.89 | 566.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 12:15:00 | 553.80 | 566.90 | 566.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 13:15:00 | 552.55 | 566.76 | 566.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 569.05 | 565.68 | 566.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-15 10:15:00 | 563.80 | 565.90 | 566.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 563.80 | 565.90 | 566.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-01-15 13:15:00 | 563.45 | 565.84 | 566.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-01-23 09:15:00 | 596.00 | 563.05 | 564.74 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 13:15:00 | 595.90 | 543.64 | 543.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 14:15:00 | 598.85 | 544.18 | 543.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 14:15:00 | 586.55 | 593.24 | 576.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 10:15:00 | 610.00 | 593.38 | 576.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-28 11:15:00 | 580.00 | 596.12 | 582.93 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 648.05 | 697.42 | 697.64 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 757.10 | 695.03 | 694.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 758.80 | 695.66 | 695.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 734.40 | 738.03 | 723.54 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 651.10 | 713.29 | 713.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 647.25 | 704.37 | 708.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 675.75 | 672.08 | 688.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 652.60 | 670.12 | 684.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 662.50 | 659.54 | 675.72 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-19 13:15:00 | 676.80 | 660.06 | 675.51 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 703.40 | 649.32 | 649.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 11:15:00 | 709.00 | 650.38 | 649.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 710.00 | 714.56 | 691.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 12:15:00 | 719.20 | 714.47 | 692.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 722.45 | 737.15 | 718.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-20 14:15:00 | 710.55 | 736.70 | 718.06 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 935.30 | 962.49 | 962.53 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 991.90 | 962.61 | 962.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 999.20 | 964.55 | 963.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 961.90 | 968.78 | 965.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-09 10:15:00 | 978.30 | 968.76 | 966.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-12 13:15:00 | 966.10 | 970.03 | 967.13 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 939.80 | 964.68 | 964.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 934.50 | 964.13 | 964.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 914.00 | 893.87 | 921.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 15:15:00 | 837.50 | 890.93 | 918.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 873.20 | 852.81 | 874.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-29 10:15:00 | 869.15 | 852.97 | 874.22 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-15 10:15:00 | 563.80 | 2024-01-16 12:15:00 | 556.16 | TARGET | 7.64 |
| SELL | 2024-01-15 13:15:00 | 563.45 | 2024-01-16 13:15:00 | 554.87 | TARGET | 8.58 |
| BUY | 2024-05-14 10:15:00 | 610.00 | 2024-05-28 11:15:00 | 580.00 | EXIT_EMA400 | -30.00 |
| SELL | 2025-02-10 09:15:00 | 652.60 | 2025-02-19 13:15:00 | 676.80 | EXIT_EMA400 | -24.20 |
| BUY | 2025-05-23 12:15:00 | 719.20 | 2025-06-19 09:15:00 | 799.62 | TARGET | 80.42 |
| BUY | 2026-02-09 10:15:00 | 978.30 | 2026-02-12 13:15:00 | 966.10 | EXIT_EMA400 | -12.20 |
| SELL | 2026-04-29 10:15:00 | 869.15 | 2026-04-29 14:15:00 | 853.94 | TARGET | 15.21 |
