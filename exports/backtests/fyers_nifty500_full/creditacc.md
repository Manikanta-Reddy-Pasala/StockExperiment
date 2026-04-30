# CreditAccess Grameen Ltd. (CREDITACC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1300.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 29.44
- **Avg P&L per closed trade:** 9.81

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 1013.80 | 962.13 | 962.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 1050.60 | 968.13 | 965.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 15:15:00 | 988.00 | 990.41 | 977.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-12 12:15:00 | 1008.55 | 990.58 | 978.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-14 12:15:00 | 977.00 | 992.83 | 980.15 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 862.30 | 970.40 | 970.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 851.30 | 968.17 | 969.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 973.40 | 959.26 | 964.77 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 998.95 | 960.10 | 959.92 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 950.60 | 959.65 | 959.69 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 970.45 | 959.74 | 959.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 988.65 | 960.02 | 959.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1103.10 | 1111.14 | 1062.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-26 09:15:00 | 1152.00 | 1111.40 | 1069.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 1119.60 | 1158.76 | 1120.12 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1255.30 | 1350.52 | 1350.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1252.70 | 1349.55 | 1350.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1313.10 | 1310.90 | 1326.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 1288.10 | 1310.02 | 1325.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 1317.50 | 1297.83 | 1315.75 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-02-12 12:15:00 | 1008.55 | 2025-02-13 14:15:00 | 1099.79 | TARGET | 91.24 |
| BUY | 2025-05-26 09:15:00 | 1152.00 | 2025-06-19 10:15:00 | 1119.60 | EXIT_EMA400 | -32.40 |
| SELL | 2025-12-26 13:15:00 | 1288.10 | 2026-01-05 11:15:00 | 1317.50 | EXIT_EMA400 | -29.40 |
