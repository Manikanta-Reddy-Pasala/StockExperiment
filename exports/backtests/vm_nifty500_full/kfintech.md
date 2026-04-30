# Kfin Technologies Ltd. (KFINTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 895.70
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

### Cycle 1 — SELL (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 09:15:00 | 1089.00 | 1208.64 | 1209.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 1061.65 | 1194.02 | 1201.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 969.50 | 958.16 | 1031.92 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 1243.50 | 1054.98 | 1054.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 1280.70 | 1059.12 | 1056.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 1098.20 | 1114.00 | 1087.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 1131.50 | 1107.40 | 1086.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1131.50 | 1107.40 | 1086.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 12:15:00 | 1081.60 | 1107.06 | 1086.98 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1101.00 | 1194.89 | 1194.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 1093.00 | 1193.88 | 1194.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1116.10 | 1110.23 | 1138.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 09:15:00 | 1096.90 | 1110.48 | 1135.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-18 09:15:00 | 1138.70 | 1109.88 | 1132.64 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-08 09:15:00 | 1131.50 | 2025-05-08 12:15:00 | 1081.60 | EXIT_EMA400 | -49.90 |
| SELL | 2025-09-15 09:15:00 | 1096.90 | 2025-09-18 09:15:00 | 1138.70 | EXIT_EMA400 | -41.80 |
