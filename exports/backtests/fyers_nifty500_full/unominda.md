# UNO Minda Ltd. (UNOMINDA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1113.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -141.00
- **Avg P&L per closed trade:** -35.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 980.00 | 1050.30 | 1050.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 969.95 | 1049.50 | 1050.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 988.30 | 980.97 | 1005.23 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 1081.15 | 1018.55 | 1018.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1091.85 | 1021.08 | 1019.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 12:15:00 | 1039.95 | 1040.18 | 1030.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 14:15:00 | 1043.10 | 1040.21 | 1030.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1033.70 | 1040.91 | 1031.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-17 12:15:00 | 1026.25 | 1040.76 | 1031.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 916.55 | 1036.73 | 1036.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 910.65 | 1035.48 | 1036.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 995.00 | 994.42 | 1013.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 966.25 | 993.99 | 1012.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 1015.30 | 992.41 | 1010.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 992.30 | 917.77 | 917.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1005.00 | 921.38 | 919.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1068.80 | 1069.13 | 1031.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 11:15:00 | 1100.50 | 1070.23 | 1035.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 10:15:00 | 1049.20 | 1076.70 | 1049.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1182.10 | 1260.87 | 1261.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1176.20 | 1260.03 | 1260.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1223.80 | 1202.80 | 1226.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 10:15:00 | 1209.10 | 1203.89 | 1225.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 12:15:00 | 1232.90 | 1203.59 | 1224.68 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-13 14:15:00 | 1043.10 | 2024-12-17 12:15:00 | 1026.25 | EXIT_EMA400 | -16.85 |
| SELL | 2025-02-03 09:15:00 | 966.25 | 2025-02-05 09:15:00 | 1015.30 | EXIT_EMA400 | -49.05 |
| BUY | 2025-07-15 11:15:00 | 1100.50 | 2025-07-29 10:15:00 | 1049.20 | EXIT_EMA400 | -51.30 |
| SELL | 2026-02-04 10:15:00 | 1209.10 | 2026-02-05 12:15:00 | 1232.90 | EXIT_EMA400 | -23.80 |
