# Max Healthcare Institute Ltd. (MAXHEALTH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 992.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 138.70
- **Avg P&L per closed trade:** 34.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 1006.90 | 1072.65 | 1072.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 985.85 | 1048.90 | 1059.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 15:15:00 | 1017.50 | 1017.12 | 1036.75 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 1173.50 | 1053.65 | 1053.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1176.10 | 1111.78 | 1093.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1133.70 | 1140.36 | 1116.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 12:15:00 | 1156.30 | 1139.94 | 1118.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1220.90 | 1236.83 | 1202.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-21 12:15:00 | 1224.90 | 1236.37 | 1203.25 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-14 09:15:00 | 1227.40 | 1251.91 | 1228.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 1167.90 | 1215.89 | 1216.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1164.20 | 1214.93 | 1215.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1155.00 | 1152.79 | 1174.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-10 09:15:00 | 1144.40 | 1152.89 | 1173.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1166.10 | 1153.05 | 1171.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-15 11:15:00 | 1159.50 | 1153.11 | 1171.46 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-16 13:15:00 | 1172.10 | 1154.00 | 1171.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-02 12:15:00 | 1156.30 | 2025-06-26 11:15:00 | 1270.33 | TARGET | 114.03 |
| BUY | 2025-07-21 12:15:00 | 1224.90 | 2025-08-06 09:15:00 | 1289.86 | TARGET | 64.96 |
| SELL | 2025-10-10 09:15:00 | 1144.40 | 2025-10-16 13:15:00 | 1172.10 | EXIT_EMA400 | -27.70 |
| SELL | 2025-10-15 11:15:00 | 1159.50 | 2025-10-16 13:15:00 | 1172.10 | EXIT_EMA400 | -12.60 |
