# IndusInd Bank Ltd. (INDUSINDBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 919.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -7.46
- **Avg P&L per closed trade:** -1.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 1469.75 | 1416.18 | 1415.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 13:15:00 | 1472.00 | 1417.74 | 1416.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 1434.55 | 1437.01 | 1427.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-25 14:15:00 | 1440.85 | 1437.04 | 1428.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1441.15 | 1439.90 | 1430.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-01 11:15:00 | 1428.70 | 1439.70 | 1430.50 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 10:15:00 | 1365.35 | 1422.46 | 1422.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 13:15:00 | 1349.00 | 1420.49 | 1421.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.56 | 1068.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 978.50 | 993.04 | 1065.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-01 09:15:00 | 1018.75 | 970.01 | 1014.00 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.17 | 825.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.17 | 839.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 13:15:00 | 852.95 | 850.09 | 839.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 795.95 | 833.84 | 833.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.59 | 828.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.34 | 770.40 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.97 | 771.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 820.50 | 772.97 | 772.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 832.00 | 833.00 | 813.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 09:15:00 | 838.20 | 833.05 | 813.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.15 | 862.21 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-27 14:15:00 | 894.60 | 888.14 | 862.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 900.85 | 890.08 | 866.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-02 14:15:00 | 911.00 | 890.55 | 868.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 882.85 | 924.05 | 903.16 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 820.75 | 888.47 | 888.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 814.65 | 885.78 | 887.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 12:15:00 | 831.40 | 835.23 | 856.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-25 14:15:00 | 1440.85 | 2024-09-27 09:15:00 | 1479.36 | TARGET | 38.51 |
| SELL | 2025-01-06 10:15:00 | 978.50 | 2025-02-01 09:15:00 | 1018.75 | EXIT_EMA400 | -40.25 |
| BUY | 2025-07-21 13:15:00 | 852.95 | 2025-07-25 09:15:00 | 835.75 | EXIT_EMA400 | -17.20 |
| BUY | 2025-12-11 09:15:00 | 838.20 | 2026-01-06 09:15:00 | 911.33 | TARGET | 73.13 |
| BUY | 2026-01-27 14:15:00 | 894.60 | 2026-03-09 09:15:00 | 882.85 | EXIT_EMA400 | -11.75 |
| BUY | 2026-02-02 14:15:00 | 911.00 | 2026-03-09 09:15:00 | 882.85 | EXIT_EMA400 | -28.15 |
| SELL | 2026-04-08 12:15:00 | 831.40 | 2026-04-16 09:15:00 | 853.15 | EXIT_EMA400 | -21.75 |
