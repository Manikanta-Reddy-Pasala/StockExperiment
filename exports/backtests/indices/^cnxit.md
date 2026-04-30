# NIFTY IT (^CNXIT)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5015 bars)
- **Last close:** 29353.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 5000 pts (index)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -2708.30
- **Avg P&L per closed trade:** -386.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 30811.00 | 31358.69 | 31358.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 30717.40 | 31244.38 | 31297.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 31313.30 | 31159.32 | 31249.00 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 11:15:00 | 32379.80 | 31331.72 | 31329.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 32483.80 | 31383.64 | 31355.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 37313.95 | 37357.55 | 36343.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 14:15:00 | 37691.35 | 37228.10 | 36500.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-19 09:15:00 | 36263.70 | 37224.38 | 36554.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 35232.95 | 36120.94 | 36124.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 14:15:00 | 35070.35 | 36051.17 | 36088.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 33810.75 | 33798.65 | 34472.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 14:15:00 | 33543.00 | 33831.01 | 34387.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 35089.80 | 33531.12 | 34107.48 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 12:15:00 | 35959.10 | 34451.45 | 34444.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 36124.30 | 34483.49 | 34460.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 38121.95 | 38700.57 | 37290.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 39199.70 | 38687.55 | 37325.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 14:15:00 | 41159.85 | 42133.69 | 41383.56 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 14:15:00 | 42648.40 | 43111.31 | 43112.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 11:15:00 | 42339.75 | 43089.68 | 43101.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 42997.25 | 42965.59 | 43034.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 13:15:00 | 42892.35 | 42964.86 | 43033.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 42996.35 | 42963.39 | 43031.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-06 10:15:00 | 42928.95 | 42963.04 | 43031.10 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 43007.90 | 42962.07 | 43029.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-06 15:15:00 | 43029.60 | 42962.75 | 43029.26 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 12:15:00 | 38502.95 | 37331.25 | 37327.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 38933.15 | 37393.52 | 37359.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 38448.35 | 38463.16 | 38073.40 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 36150.60 | 37826.48 | 37828.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 35957.60 | 37742.98 | 37785.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 35775.90 | 35709.88 | 36443.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 09:15:00 | 35522.95 | 35711.26 | 36422.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-25 10:15:00 | 36411.30 | 35711.08 | 36394.47 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 36351.15 | 35675.70 | 35673.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 36388.50 | 35736.18 | 35704.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 37845.80 | 37967.23 | 37285.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-01 11:15:00 | 38196.90 | 37968.38 | 37316.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 37778.75 | 38139.21 | 37643.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-21 10:15:00 | 37629.15 | 38134.13 | 37643.13 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 35452.05 | 37472.16 | 37480.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 35348.50 | 37451.03 | 37469.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 30647.15 | 30614.69 | 32405.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 11:15:00 | 30217.95 | 30987.41 | 32067.56 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-14 14:15:00 | 37691.35 | 2024-03-19 09:15:00 | 36263.70 | EXIT_EMA400 | -1427.65 |
| SELL | 2024-05-29 14:15:00 | 33543.00 | 2024-06-07 09:15:00 | 35089.80 | EXIT_EMA400 | -1546.80 |
| BUY | 2024-08-06 09:15:00 | 39199.70 | 2024-10-22 14:15:00 | 41159.85 | EXIT_EMA400 | 1960.15 |
| SELL | 2025-02-05 13:15:00 | 42892.35 | 2025-02-06 15:15:00 | 43029.60 | EXIT_EMA400 | -137.25 |
| SELL | 2025-02-06 10:15:00 | 42928.95 | 2025-02-06 15:15:00 | 43029.60 | EXIT_EMA400 | -100.65 |
| SELL | 2025-08-22 09:15:00 | 35522.95 | 2025-08-25 10:15:00 | 36411.30 | EXIT_EMA400 | -888.35 |
| BUY | 2026-01-01 11:15:00 | 38196.90 | 2026-01-21 10:15:00 | 37629.15 | EXIT_EMA400 | -567.75 |
