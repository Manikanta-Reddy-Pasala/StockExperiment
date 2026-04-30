# Kfin Technologies Ltd. (KFINTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 898.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -91.70
- **Avg P&L per closed trade:** -45.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1052.95 | 1209.61 | 1209.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 14:15:00 | 1038.40 | 1184.56 | 1195.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 969.95 | 957.29 | 1030.70 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1247.00 | 1052.71 | 1052.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 1282.10 | 1058.77 | 1055.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 1098.00 | 1113.68 | 1087.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 1131.50 | 1107.12 | 1086.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1131.50 | 1107.12 | 1086.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 12:15:00 | 1081.60 | 1106.78 | 1086.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1093.00 | 1193.90 | 1194.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1081.00 | 1192.78 | 1193.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1116.10 | 1110.36 | 1138.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 09:15:00 | 1096.90 | 1110.59 | 1135.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-18 09:15:00 | 1138.70 | 1109.95 | 1132.69 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-08 09:15:00 | 1131.50 | 2025-05-08 12:15:00 | 1081.60 | EXIT_EMA400 | -49.90 |
| SELL | 2025-09-15 09:15:00 | 1096.90 | 2025-09-18 09:15:00 | 1138.70 | EXIT_EMA400 | -41.80 |
