# UTI Asset Management Company Ltd. (UTIAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 950.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 36.15
- **Avg P&L per closed trade:** 12.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1221.25 | 1273.44 | 1273.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 1214.55 | 1272.32 | 1273.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 987.50 | 984.19 | 1053.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-07 09:15:00 | 976.00 | 1041.59 | 1050.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 14:15:00 | 1045.00 | 1032.57 | 1044.93 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1175.40 | 1055.16 | 1054.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 12:15:00 | 1189.00 | 1057.64 | 1056.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 1341.40 | 1350.25 | 1275.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 14:15:00 | 1355.10 | 1350.12 | 1276.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 11:15:00 | 1312.40 | 1348.57 | 1312.89 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 1243.50 | 1325.78 | 1325.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1236.00 | 1319.56 | 1322.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 1140.50 | 1139.29 | 1176.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 09:15:00 | 1127.20 | 1139.12 | 1176.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 1106.40 | 1058.80 | 1099.40 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-05-07 09:15:00 | 976.00 | 2025-05-12 14:15:00 | 1045.00 | EXIT_EMA400 | -69.00 |
| BUY | 2025-07-25 14:15:00 | 1355.10 | 2025-08-28 11:15:00 | 1312.40 | EXIT_EMA400 | -42.70 |
| SELL | 2026-01-06 09:15:00 | 1127.20 | 2026-01-23 13:15:00 | 979.35 | TARGET | 147.85 |
