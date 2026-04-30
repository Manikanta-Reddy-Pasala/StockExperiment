# Chalet Hotels Ltd. (CHALET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 756.00
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
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -135.20
- **Avg P&L per closed trade:** -27.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 785.10 | 815.77 | 815.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 15:15:00 | 784.00 | 815.45 | 815.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 801.75 | 798.90 | 805.93 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 875.80 | 811.17 | 811.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 881.45 | 813.13 | 812.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 849.45 | 851.96 | 835.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 13:15:00 | 874.40 | 852.31 | 836.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 10:15:00 | 848.45 | 867.76 | 850.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 784.10 | 904.43 | 904.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 760.75 | 883.13 | 893.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 754.25 | 746.20 | 792.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-03 09:15:00 | 721.70 | 745.96 | 792.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-07 14:15:00 | 790.05 | 750.17 | 787.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 863.00 | 802.44 | 802.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 873.95 | 803.15 | 802.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 788.75 | 805.71 | 803.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 14:15:00 | 815.00 | 805.27 | 803.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 815.00 | 805.27 | 803.77 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 09:15:00 | 822.10 | 805.49 | 803.90 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 810.55 | 806.57 | 804.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-09 10:15:00 | 802.00 | 806.52 | 804.49 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 924.10 | 960.31 | 960.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 909.05 | 958.86 | 959.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 908.75 | 908.42 | 926.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 902.20 | 908.35 | 925.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 896.05 | 884.78 | 901.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 910.00 | 885.04 | 901.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-19 13:15:00 | 874.40 | 2024-10-07 10:15:00 | 848.45 | EXIT_EMA400 | -25.95 |
| SELL | 2025-03-03 09:15:00 | 721.70 | 2025-03-07 14:15:00 | 790.05 | EXIT_EMA400 | -68.35 |
| BUY | 2025-04-07 14:15:00 | 815.00 | 2025-04-09 10:15:00 | 802.00 | EXIT_EMA400 | -13.00 |
| BUY | 2025-04-08 09:15:00 | 822.10 | 2025-04-09 10:15:00 | 802.00 | EXIT_EMA400 | -20.10 |
| SELL | 2025-12-04 09:15:00 | 902.20 | 2026-01-02 09:15:00 | 910.00 | EXIT_EMA400 | -7.80 |
