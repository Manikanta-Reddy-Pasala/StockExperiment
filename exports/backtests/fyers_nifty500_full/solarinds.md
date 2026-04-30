# Solar Industries India Ltd. (SOLARINDS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 15470.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 1773.57
- **Avg P&L per closed trade:** 354.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 9895.00 | 10795.14 | 10797.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 9724.60 | 10784.49 | 10792.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 10774.35 | 10321.99 | 10496.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 09:15:00 | 10189.55 | 10517.83 | 10555.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 10018.65 | 9673.27 | 9942.05 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 10717.95 | 9615.68 | 9611.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 10763.00 | 9627.09 | 9617.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 16437.00 | 16508.38 | 15298.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-09 09:15:00 | 16566.00 | 16506.48 | 15333.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 15407.00 | 16425.78 | 15398.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 09:15:00 | 15270.00 | 16404.19 | 15397.58 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 14535.00 | 15004.74 | 15006.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 14470.00 | 14999.42 | 15004.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 14575.00 | 14409.16 | 14639.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 12:15:00 | 14130.00 | 14472.83 | 14624.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 14116.00 | 13928.55 | 14130.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-13 14:15:00 | 13741.00 | 13934.69 | 14122.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 10:15:00 | 13040.00 | 12629.50 | 13039.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 13241.00 | 13128.09 | 13127.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 13288.00 | 13132.72 | 13130.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 13825.00 | 13894.43 | 13596.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 13:15:00 | 13984.00 | 13896.17 | 13601.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-19 09:15:00 | 13329.00 | 13885.90 | 13610.66 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 12529.00 | 13399.38 | 13402.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 12456.00 | 13365.29 | 13384.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 13330.00 | 13329.19 | 13364.77 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 14114.00 | 13397.06 | 13395.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 14510.00 | 13475.13 | 13435.99 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-19 09:15:00 | 10189.55 | 2025-01-13 11:15:00 | 9093.10 | TARGET | 1096.45 |
| BUY | 2025-07-09 09:15:00 | 16566.00 | 2025-07-14 09:15:00 | 15270.00 | EXIT_EMA400 | -1296.00 |
| SELL | 2025-09-23 12:15:00 | 14130.00 | 2025-12-08 09:15:00 | 12645.87 | TARGET | 1484.13 |
| SELL | 2025-11-13 14:15:00 | 13741.00 | 2025-12-08 09:15:00 | 12597.02 | TARGET | 1143.98 |
| BUY | 2026-03-17 13:15:00 | 13984.00 | 2026-03-19 09:15:00 | 13329.00 | EXIT_EMA400 | -655.00 |
