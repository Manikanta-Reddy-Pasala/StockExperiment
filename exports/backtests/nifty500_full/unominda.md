# UNO Minda Ltd. (UNOMINDA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4997 bars)
- **Last close:** 1112.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 131.76
- **Avg P&L per closed trade:** 14.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 13:15:00 | 572.25 | 597.82 | 597.92 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 637.05 | 596.82 | 596.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 641.70 | 600.21 | 598.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 10:15:00 | 642.30 | 644.03 | 628.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-13 13:15:00 | 648.90 | 643.72 | 628.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 649.45 | 648.27 | 633.71 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-21 10:15:00 | 650.90 | 648.30 | 633.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 673.45 | 685.34 | 671.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-08 13:15:00 | 665.00 | 684.89 | 671.14 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 635.60 | 660.96 | 661.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 13:15:00 | 627.25 | 656.48 | 658.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 15:15:00 | 655.00 | 653.61 | 656.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-11 09:15:00 | 634.15 | 653.41 | 656.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-21 14:15:00 | 650.05 | 641.98 | 649.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 09:15:00 | 692.70 | 654.93 | 654.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 12:15:00 | 697.45 | 656.12 | 655.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 1035.15 | 1037.64 | 943.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 12:15:00 | 1052.90 | 1024.35 | 963.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 983.90 | 1028.70 | 975.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 962.80 | 1028.04 | 975.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 965.00 | 1048.76 | 1049.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 961.95 | 1047.90 | 1048.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 989.00 | 981.15 | 1005.20 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 1081.35 | 1018.53 | 1018.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1092.10 | 1021.07 | 1019.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 12:15:00 | 1039.80 | 1040.18 | 1030.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 14:15:00 | 1043.10 | 1040.21 | 1030.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1033.70 | 1040.92 | 1031.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-17 12:15:00 | 1026.25 | 1040.77 | 1031.73 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 916.65 | 1036.74 | 1036.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 910.70 | 1035.48 | 1036.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 996.35 | 993.56 | 1012.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 14:15:00 | 986.40 | 1005.09 | 1015.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1011.40 | 1004.59 | 1014.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-13 09:15:00 | 971.00 | 1003.71 | 1014.20 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-17 09:15:00 | 938.00 | 894.46 | 936.28 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 985.20 | 918.53 | 918.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1005.00 | 921.48 | 919.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1068.80 | 1069.18 | 1031.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 11:15:00 | 1100.50 | 1070.25 | 1035.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 10:15:00 | 1049.20 | 1076.70 | 1049.75 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1182.10 | 1260.99 | 1261.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1176.20 | 1260.15 | 1260.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1222.80 | 1208.60 | 1230.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 15:15:00 | 1211.00 | 1209.12 | 1230.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 12:15:00 | 1233.70 | 1208.51 | 1228.85 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 10:15:00 | 650.90 | 2024-01-05 15:15:00 | 702.21 | TARGET | 51.31 |
| BUY | 2023-12-13 13:15:00 | 648.90 | 2024-01-08 09:15:00 | 709.72 | TARGET | 60.82 |
| SELL | 2024-03-11 09:15:00 | 634.15 | 2024-03-21 14:15:00 | 650.05 | EXIT_EMA400 | -15.90 |
| BUY | 2024-07-29 12:15:00 | 1052.90 | 2024-08-05 10:15:00 | 962.80 | EXIT_EMA400 | -90.10 |
| BUY | 2024-12-13 14:15:00 | 1043.10 | 2024-12-17 12:15:00 | 1026.25 | EXIT_EMA400 | -16.85 |
| SELL | 2025-02-11 14:15:00 | 986.40 | 2025-02-14 10:15:00 | 899.52 | TARGET | 86.88 |
| SELL | 2025-02-13 09:15:00 | 971.00 | 2025-02-20 09:15:00 | 841.40 | TARGET | 129.60 |
| BUY | 2025-07-15 11:15:00 | 1100.50 | 2025-07-29 10:15:00 | 1049.20 | EXIT_EMA400 | -51.30 |
| SELL | 2026-02-03 15:15:00 | 1211.00 | 2026-02-05 12:15:00 | 1233.70 | EXIT_EMA400 | -22.70 |
