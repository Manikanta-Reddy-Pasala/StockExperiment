# Adani Green Energy Ltd. (ADANIGREEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1231.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -64.48
- **Avg P&L per closed trade:** -10.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1742.55 | 1852.05 | 1852.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1734.80 | 1849.81 | 1850.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 888.45 | 882.61 | 978.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-09 09:15:00 | 853.45 | 909.19 | 953.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 952.85 | 907.70 | 947.93 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1002.25 | 953.13 | 952.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 1017.90 | 953.77 | 953.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 989.00 | 995.99 | 979.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 1000.10 | 985.42 | 977.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 978.90 | 985.42 | 977.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-26 12:15:00 | 986.90 | 985.42 | 977.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-08 11:15:00 | 984.30 | 995.32 | 985.34 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 910.05 | 991.45 | 991.65 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1148.65 | 975.00 | 974.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 1158.00 | 976.82 | 975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1030.03 | 1010.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-14 14:15:00 | 1036.30 | 1030.05 | 1010.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.45 | 1017.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-27 11:15:00 | 1022.60 | 1034.33 | 1017.09 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1017.20 | 1034.02 | 1017.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.69 | 1017.11 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1050.20 | 1034.11 | 1034.09 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1034.03 | 1034.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1033.91 | 1034.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 1012.30 | 1025.16 | 1028.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 966.60 | 920.08 | 960.41 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.57 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-09 09:15:00 | 853.45 | 2025-04-16 09:15:00 | 952.85 | EXIT_EMA400 | -99.40 |
| BUY | 2025-06-26 12:15:00 | 986.90 | 2025-06-27 09:15:00 | 1014.59 | TARGET | 27.69 |
| BUY | 2025-06-24 11:15:00 | 1000.10 | 2025-07-08 11:15:00 | 984.30 | EXIT_EMA400 | -15.80 |
| BUY | 2025-10-14 14:15:00 | 1036.30 | 2025-10-27 15:15:00 | 1017.00 | EXIT_EMA400 | -19.30 |
| BUY | 2025-10-27 11:15:00 | 1022.60 | 2025-10-27 15:15:00 | 1017.00 | EXIT_EMA400 | -5.60 |
| SELL | 2026-01-08 09:15:00 | 1012.30 | 2026-01-09 12:15:00 | 964.36 | TARGET | 47.94 |
