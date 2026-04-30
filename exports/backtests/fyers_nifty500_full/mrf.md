# MRF Ltd. (MRF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 130000.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 9287.68
- **Avg P&L per closed trade:** 2321.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 129758.05 | 134543.56 | 134545.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 128999.00 | 134309.07 | 134426.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 125491.75 | 124720.33 | 127842.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 11:15:00 | 125159.70 | 124734.04 | 127818.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 126858.00 | 124838.75 | 127149.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 09:15:00 | 129421.00 | 124884.34 | 127160.59 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 130000.00 | 128675.15 | 128675.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 130900.00 | 128714.85 | 128694.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 128314.00 | 129243.65 | 128989.46 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 122994.70 | 128715.08 | 128742.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 121587.40 | 128583.28 | 128675.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 108800.05 | 108676.32 | 112644.72 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 126500.00 | 113958.76 | 113920.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 126740.00 | 114210.17 | 114047.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 136715.00 | 136842.67 | 130512.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 09:15:00 | 138555.00 | 136838.28 | 130666.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 135000.00 | 137172.95 | 132971.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-24 11:15:00 | 136900.00 | 137017.14 | 133115.34 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 14:15:00 | 142500.00 | 145991.33 | 142574.16 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 148510.00 | 153067.73 | 153080.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 148100.00 | 151991.20 | 152478.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 136315.00 | 144203.07 | 145713.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 136790.00 | 133733.81 | 137491.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 14:15:00 | 138065.00 | 133903.44 | 137484.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 11:15:00 | 125159.70 | 2024-12-05 09:15:00 | 129421.00 | EXIT_EMA400 | -4261.30 |
| BUY | 2025-06-24 11:15:00 | 136900.00 | 2025-07-09 11:15:00 | 148253.98 | TARGET | 11353.98 |
| BUY | 2025-06-05 09:15:00 | 138555.00 | 2025-08-08 14:15:00 | 142500.00 | EXIT_EMA400 | 3945.00 |
| SELL | 2026-03-04 09:15:00 | 136315.00 | 2026-04-15 14:15:00 | 138065.00 | EXIT_EMA400 | -1750.00 |
