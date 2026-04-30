# Solar Industries India Ltd. (SOLARINDS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 15439.00
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
- **Total realized P&L (per unit):** 1773.91
- **Avg P&L per closed trade:** 354.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 9718.50 | 10785.29 | 10785.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 9661.35 | 10532.33 | 10645.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 10772.30 | 10326.81 | 10496.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 09:15:00 | 10189.55 | 10521.21 | 10555.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 10014.85 | 9673.55 | 9942.27 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 10717.95 | 9612.84 | 9609.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 10763.00 | 9624.28 | 9614.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 16437.00 | 16508.48 | 15298.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-09 09:15:00 | 16566.00 | 16506.60 | 15333.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 15407.00 | 16425.61 | 15398.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 09:15:00 | 15270.00 | 16404.01 | 15397.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 14645.00 | 15008.55 | 15008.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 14535.00 | 15003.84 | 15006.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 14575.00 | 14407.97 | 14638.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 12:15:00 | 14130.00 | 14472.31 | 14624.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 14116.00 | 13928.23 | 14130.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-13 14:15:00 | 13741.00 | 13934.41 | 14122.13 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 10:15:00 | 13040.00 | 12626.78 | 13038.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 13384.00 | 13124.78 | 13124.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 13460.00 | 13151.62 | 13138.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 13825.00 | 13889.68 | 13590.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 13:15:00 | 13984.00 | 13891.56 | 13596.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-19 09:15:00 | 13329.00 | 13881.79 | 13605.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 12530.00 | 13397.37 | 13398.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 12456.00 | 13362.93 | 13380.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 13330.00 | 13327.73 | 13361.53 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 14114.00 | 13395.62 | 13392.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 14510.00 | 13473.91 | 13433.15 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-19 09:15:00 | 10189.55 | 2025-01-13 11:15:00 | 9090.72 | TARGET | 1098.83 |
| BUY | 2025-07-09 09:15:00 | 16566.00 | 2025-07-14 09:15:00 | 15270.00 | EXIT_EMA400 | -1296.00 |
| SELL | 2025-09-23 12:15:00 | 14130.00 | 2025-12-08 09:15:00 | 12647.31 | TARGET | 1482.69 |
| SELL | 2025-11-13 14:15:00 | 13741.00 | 2025-12-08 09:15:00 | 12597.61 | TARGET | 1143.39 |
| BUY | 2026-03-17 13:15:00 | 13984.00 | 2026-03-19 09:15:00 | 13329.00 | EXIT_EMA400 | -655.00 |
