# Cohance Lifesciences Ltd. (COHANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 480.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -154.25
- **Avg P&L per closed trade:** -30.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 1114.25 | 1213.12 | 1213.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 1095.20 | 1200.43 | 1206.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 1126.60 | 1077.62 | 1122.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 1048.90 | 1095.35 | 1122.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-13 09:15:00 | 1145.00 | 1093.33 | 1120.81 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1177.50 | 1136.28 | 1136.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1182.85 | 1136.74 | 1136.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 1138.60 | 1143.28 | 1140.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-12 13:15:00 | 1152.00 | 1143.42 | 1140.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-03-13 11:15:00 | 1134.70 | 1143.66 | 1140.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 11:15:00 | 1100.05 | 1139.96 | 1139.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 13:15:00 | 1093.10 | 1139.08 | 1139.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 10:15:00 | 1121.20 | 1120.65 | 1129.47 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 15:15:00 | 1201.00 | 1137.28 | 1137.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1243.20 | 1138.33 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 14:15:00 | 1144.20 | 1152.79 | 1145.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 09:15:00 | 1163.20 | 1152.81 | 1145.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1163.20 | 1152.81 | 1145.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-28 14:15:00 | 1142.00 | 1152.58 | 1145.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1059.20 | 1140.22 | 1140.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 1044.00 | 1115.21 | 1126.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 1109.30 | 1100.86 | 1115.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-29 09:15:00 | 1067.20 | 1100.85 | 1115.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1040.70 | 1011.55 | 1045.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-07 09:15:00 | 1024.45 | 1012.09 | 1045.28 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-15 09:15:00 | 1055.65 | 1010.66 | 1038.13 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-12 09:15:00 | 1048.90 | 2025-02-13 09:15:00 | 1145.00 | EXIT_EMA400 | -96.10 |
| BUY | 2025-03-12 13:15:00 | 1152.00 | 2025-03-13 11:15:00 | 1134.70 | EXIT_EMA400 | -17.30 |
| BUY | 2025-04-28 09:15:00 | 1163.20 | 2025-04-28 14:15:00 | 1142.00 | EXIT_EMA400 | -21.20 |
| SELL | 2025-05-29 09:15:00 | 1067.20 | 2025-07-15 09:15:00 | 1055.65 | EXIT_EMA400 | 11.55 |
| SELL | 2025-07-07 09:15:00 | 1024.45 | 2025-07-15 09:15:00 | 1055.65 | EXIT_EMA400 | -31.20 |
