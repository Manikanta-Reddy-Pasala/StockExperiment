# Asahi India Glass Ltd. (ASAHIINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 836.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 61.33
- **Avg P&L per closed trade:** 15.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 656.65 | 698.63 | 698.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 648.05 | 697.71 | 698.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 687.45 | 686.22 | 691.55 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 758.80 | 695.76 | 695.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 11:15:00 | 763.10 | 696.43 | 695.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 734.40 | 737.94 | 723.65 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 651.10 | 713.14 | 713.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 647.25 | 704.29 | 708.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 675.75 | 671.99 | 688.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 15:15:00 | 655.35 | 671.39 | 687.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 662.10 | 659.26 | 675.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-19 13:15:00 | 676.80 | 659.75 | 674.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 703.40 | 649.19 | 649.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 11:15:00 | 709.00 | 650.25 | 649.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 710.00 | 714.50 | 691.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 12:15:00 | 719.20 | 714.36 | 692.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 722.45 | 737.13 | 718.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-20 14:15:00 | 710.75 | 736.68 | 717.99 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 935.30 | 962.19 | 962.32 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 981.90 | 962.53 | 962.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 993.90 | 963.07 | 962.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-09 10:15:00 | 978.30 | 970.01 | 966.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-12 13:15:00 | 966.10 | 970.98 | 967.78 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 934.50 | 964.87 | 965.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 931.20 | 964.03 | 964.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 914.00 | 894.04 | 921.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 15:15:00 | 837.50 | 891.07 | 918.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 873.20 | 852.77 | 873.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-29 10:15:00 | 869.15 | 852.93 | 873.84 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-01 15:15:00 | 655.35 | 2025-02-19 13:15:00 | 676.80 | EXIT_EMA400 | -21.45 |
| BUY | 2025-05-23 12:15:00 | 719.20 | 2025-06-19 09:15:00 | 800.12 | TARGET | 80.92 |
| BUY | 2026-02-09 10:15:00 | 978.30 | 2026-02-12 13:15:00 | 966.10 | EXIT_EMA400 | -12.20 |
| SELL | 2026-04-29 10:15:00 | 869.15 | 2026-04-29 14:15:00 | 855.08 | TARGET | 14.07 |
