# Tata Technologies Ltd. (TATATECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 581.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -41.86
- **Avg P&L per closed trade:** -6.98

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 15:15:00 | 1098.50 | 1014.19 | 1014.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 1109.75 | 1015.14 | 1014.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 1057.25 | 1060.60 | 1043.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 09:15:00 | 1082.70 | 1060.80 | 1044.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1064.45 | 1077.73 | 1058.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 11:15:00 | 1076.90 | 1077.66 | 1058.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-04 14:15:00 | 1056.65 | 1077.23 | 1058.79 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 13:15:00 | 996.60 | 1051.36 | 1051.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 974.05 | 1030.29 | 1039.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 702.00 | 696.89 | 753.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-28 15:15:00 | 675.00 | 695.92 | 742.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 709.05 | 670.63 | 710.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 09:15:00 | 712.10 | 671.72 | 710.26 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 775.50 | 713.30 | 713.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 778.70 | 722.84 | 718.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 751.95 | 747.73 | 734.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-18 10:15:00 | 734.95 | 747.30 | 735.25 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 705.10 | 727.03 | 727.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 703.25 | 723.67 | 725.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-18 14:15:00 | 713.80 | 720.72 | 722.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 718.70 | 719.39 | 721.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-25 09:15:00 | 711.50 | 719.29 | 721.86 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 700.00 | 683.61 | 694.09 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-20 09:15:00 | 1082.70 | 2024-10-04 14:15:00 | 1056.65 | EXIT_EMA400 | -26.05 |
| BUY | 2024-10-04 11:15:00 | 1076.90 | 2024-10-04 14:15:00 | 1056.65 | EXIT_EMA400 | -20.25 |
| SELL | 2025-03-28 15:15:00 | 675.00 | 2025-04-23 09:15:00 | 712.10 | EXIT_EMA400 | -37.10 |
| BUY | 2025-06-16 13:15:00 | 751.95 | 2025-06-18 10:15:00 | 734.95 | EXIT_EMA400 | -17.00 |
| SELL | 2025-07-18 14:15:00 | 713.80 | 2025-08-04 09:15:00 | 686.35 | TARGET | 27.45 |
| SELL | 2025-07-25 09:15:00 | 711.50 | 2025-08-06 09:15:00 | 680.41 | TARGET | 31.09 |
