# Kirloskar Oil Eng Ltd. (KIRLOSENG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1698.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 95.01
- **Avg P&L per closed trade:** 23.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 1186.25 | 1258.76 | 1258.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 1174.40 | 1255.72 | 1257.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 1226.95 | 1217.61 | 1234.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-16 12:15:00 | 1210.00 | 1217.71 | 1234.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1209.30 | 1216.59 | 1233.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-17 14:15:00 | 1205.80 | 1216.49 | 1233.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1175.75 | 1156.03 | 1190.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 12:15:00 | 1191.50 | 1156.38 | 1190.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 871.75 | 764.62 | 764.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 10:15:00 | 891.35 | 797.18 | 782.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 848.00 | 849.56 | 818.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 884.75 | 850.69 | 820.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 833.85 | 851.06 | 833.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-09 09:15:00 | 839.10 | 850.45 | 833.44 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 833.85 | 849.74 | 833.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-09 15:15:00 | 831.85 | 849.57 | 833.49 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-16 12:15:00 | 1210.00 | 2024-10-22 09:15:00 | 1136.68 | TARGET | 73.32 |
| SELL | 2024-10-17 14:15:00 | 1205.80 | 2024-10-22 09:15:00 | 1123.96 | TARGET | 81.84 |
| BUY | 2025-06-17 09:15:00 | 884.75 | 2025-07-09 15:15:00 | 831.85 | EXIT_EMA400 | -52.90 |
| BUY | 2025-07-09 09:15:00 | 839.10 | 2025-07-09 15:15:00 | 831.85 | EXIT_EMA400 | -7.25 |
