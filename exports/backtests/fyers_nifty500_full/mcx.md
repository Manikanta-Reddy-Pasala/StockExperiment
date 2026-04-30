# Multi Commodity Exchange of India Ltd. (MCX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2969.00
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
- **Total realized P&L (per unit):** 374.61
- **Avg P&L per closed trade:** 124.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 1198.80 | 1225.69 | 1225.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1116.40 | 1223.32 | 1224.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1221.01 | 1175.17 | 1194.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 1178.60 | 1181.91 | 1196.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1178.60 | 1181.91 | 1196.48 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 11:15:00 | 1162.46 | 1181.70 | 1196.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1087.96 | 1037.99 | 1090.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 14:15:00 | 1092.72 | 1040.37 | 1090.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 1245.00 | 1096.68 | 1096.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1248.30 | 1114.47 | 1105.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1137.30 | 1159.07 | 1132.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 1206.50 | 1157.92 | 1133.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 10:15:00 | 1525.70 | 1625.73 | 1536.30 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 1178.60 | 2025-02-11 09:15:00 | 1124.95 | TARGET | 53.65 |
| SELL | 2025-02-10 11:15:00 | 1162.46 | 2025-02-17 09:15:00 | 1061.14 | TARGET | 101.32 |
| BUY | 2025-05-12 09:15:00 | 1206.50 | 2025-06-05 15:15:00 | 1426.15 | TARGET | 219.65 |
