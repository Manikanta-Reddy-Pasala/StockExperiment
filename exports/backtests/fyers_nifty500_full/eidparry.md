# E.I.D. Parry (India) Ltd. (EIDPARRY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 848.00
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
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 33.15
- **Avg P&L per closed trade:** 6.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 796.45 | 857.23 | 857.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 785.15 | 856.51 | 857.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 843.30 | 840.44 | 847.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 10:15:00 | 835.30 | 840.41 | 847.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-06 09:15:00 | 858.45 | 840.40 | 847.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 859.65 | 778.54 | 778.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 868.40 | 788.00 | 783.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 948.90 | 949.89 | 905.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 13:15:00 | 954.55 | 949.94 | 906.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1097.00 | 1146.07 | 1079.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-11 09:15:00 | 1137.00 | 1145.98 | 1080.28 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1125.40 | 1153.98 | 1105.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-01 13:15:00 | 1135.40 | 1149.07 | 1107.49 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 1106.50 | 1147.47 | 1108.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 1018.90 | 1093.05 | 1093.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1008.00 | 1054.83 | 1065.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-12 11:15:00 | 1039.70 | 1052.97 | 1063.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-13 09:15:00 | 1069.00 | 1052.76 | 1063.46 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-05 10:15:00 | 835.30 | 2025-02-06 09:15:00 | 858.45 | EXIT_EMA400 | -23.15 |
| BUY | 2025-06-18 13:15:00 | 954.55 | 2025-06-30 12:15:00 | 1099.55 | TARGET | 145.00 |
| BUY | 2025-08-11 09:15:00 | 1137.00 | 2025-09-02 13:15:00 | 1106.50 | EXIT_EMA400 | -30.50 |
| BUY | 2025-09-01 13:15:00 | 1135.40 | 2025-09-02 13:15:00 | 1106.50 | EXIT_EMA400 | -28.90 |
| SELL | 2025-11-12 11:15:00 | 1039.70 | 2025-11-13 09:15:00 | 1069.00 | EXIT_EMA400 | -29.30 |
