# Multi Commodity Exchange of India Ltd. (MCX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2971.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / EMA400 exits:** 3 / 0
- **Total realized P&L (per unit):** 381.81
- **Avg P&L per closed trade:** 127.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 1195.92 | 1225.34 | 1225.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1116.40 | 1223.28 | 1224.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1221.03 | 1177.05 | 1196.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 1178.60 | 1183.42 | 1197.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1178.60 | 1183.42 | 1197.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 11:15:00 | 1162.46 | 1183.18 | 1197.52 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1087.96 | 1038.24 | 1090.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 14:15:00 | 1092.73 | 1040.62 | 1090.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1227.30 | 1098.09 | 1097.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 10:15:00 | 1279.90 | 1132.30 | 1115.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1137.00 | 1159.12 | 1133.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 1206.50 | 1157.96 | 1133.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 10:15:00 | 1525.80 | 1625.70 | 1536.29 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 1178.60 | 2025-02-11 09:15:00 | 1121.05 | TARGET | 57.55 |
| SELL | 2025-02-10 11:15:00 | 1162.46 | 2025-02-17 09:15:00 | 1057.28 | TARGET | 105.18 |
| BUY | 2025-05-12 09:15:00 | 1206.50 | 2025-06-05 15:15:00 | 1425.58 | TARGET | 219.08 |
