# Oracle Financial Services Software Ltd. (OFSS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 9726.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -462.10
- **Avg P&L per closed trade:** -77.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 3916.05 | 4033.65 | 4034.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 10:15:00 | 3892.80 | 4031.02 | 4032.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 4033.50 | 4013.59 | 4023.12 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 10:15:00 | 4199.75 | 4030.72 | 4030.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 4219.35 | 4040.23 | 4035.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 13:15:00 | 4052.00 | 4060.43 | 4046.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-22 14:15:00 | 4084.95 | 4060.67 | 4046.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 4063.55 | 4062.90 | 4048.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-11-24 12:15:00 | 4045.85 | 4062.72 | 4048.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 10061.00 | 11738.15 | 11745.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 10007.55 | 11672.06 | 11712.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 10:15:00 | 7830.00 | 7829.29 | 8482.73 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 9436.00 | 8533.09 | 8531.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 9514.50 | 8656.95 | 8595.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 9010.00 | 9024.35 | 8841.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 12:15:00 | 9082.50 | 9013.61 | 8859.12 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 8852.50 | 9010.85 | 8880.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 8471.50 | 8811.51 | 8812.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 8428.00 | 8804.38 | 8809.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 8670.00 | 8663.81 | 8725.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 11:15:00 | 8615.00 | 8663.32 | 8725.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 11:15:00 | 8774.50 | 8651.45 | 8712.77 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 9120.00 | 8703.22 | 8701.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 9154.00 | 8707.70 | 8703.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 8742.50 | 8832.86 | 8775.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 12:15:00 | 8928.00 | 8762.20 | 8744.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-15 11:15:00 | 8809.00 | 8923.86 | 8841.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 8589.00 | 8784.88 | 8785.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 8558.50 | 8782.63 | 8784.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 8489.00 | 8485.53 | 8602.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-21 09:15:00 | 8272.00 | 8478.88 | 8595.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 7959.50 | 7800.30 | 8009.91 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-20 13:15:00 | 7811.50 | 7816.58 | 8005.15 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7985.00 | 7807.32 | 7991.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-23 10:15:00 | 7999.00 | 7815.83 | 7988.33 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 8117.00 | 7209.33 | 7205.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 15:15:00 | 8130.00 | 7227.34 | 7214.28 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-22 14:15:00 | 4084.95 | 2023-11-24 12:15:00 | 4045.85 | EXIT_EMA400 | -39.10 |
| BUY | 2025-07-03 12:15:00 | 9082.50 | 2025-07-10 09:15:00 | 8852.50 | EXIT_EMA400 | -230.00 |
| SELL | 2025-08-14 11:15:00 | 8615.00 | 2025-08-20 11:15:00 | 8774.50 | EXIT_EMA400 | -159.50 |
| BUY | 2025-10-03 12:15:00 | 8928.00 | 2025-10-15 11:15:00 | 8809.00 | EXIT_EMA400 | -119.00 |
| SELL | 2025-11-21 09:15:00 | 8272.00 | 2026-01-23 10:15:00 | 7999.00 | EXIT_EMA400 | 273.00 |
| SELL | 2026-01-20 13:15:00 | 7811.50 | 2026-01-23 10:15:00 | 7999.00 | EXIT_EMA400 | -187.50 |
