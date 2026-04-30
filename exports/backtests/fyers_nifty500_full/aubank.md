# AU Small Finance Bank Ltd. (AUBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1013.35
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
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -109.10
- **Avg P&L per closed trade:** -27.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 631.70 | 646.43 | 646.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 628.30 | 645.44 | 646.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 635.00 | 632.37 | 638.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-23 11:15:00 | 629.20 | 632.32 | 638.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 632.55 | 632.13 | 637.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-28 15:15:00 | 628.30 | 632.47 | 637.44 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 635.00 | 632.49 | 637.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-29 11:15:00 | 649.00 | 632.66 | 637.47 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 677.00 | 641.92 | 641.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 10:15:00 | 681.95 | 642.61 | 642.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 13:15:00 | 707.65 | 712.96 | 690.92 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 621.05 | 680.95 | 681.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 615.90 | 679.11 | 680.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 587.75 | 577.61 | 602.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 09:15:00 | 545.05 | 582.38 | 588.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 552.75 | 537.97 | 554.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 10:15:00 | 557.05 | 538.91 | 554.72 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 616.00 | 560.05 | 559.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 09:15:00 | 647.10 | 560.92 | 560.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 790.55 | 791.71 | 748.64 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 703.30 | 742.58 | 742.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 700.30 | 742.16 | 742.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 729.45 | 722.31 | 729.93 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 768.20 | 734.57 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 772.40 | 742.13 | 738.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 977.20 | 978.09 | 937.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 09:15:00 | 1006.65 | 977.89 | 939.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-27 11:15:00 | 950.05 | 984.00 | 950.59 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 883.35 | 962.32 | 962.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 881.50 | 960.80 | 961.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 941.05 | 911.05 | 931.08 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1035.05 | 945.53 | 945.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 1038.50 | 946.46 | 945.79 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-23 11:15:00 | 629.20 | 2024-08-29 11:15:00 | 649.00 | EXIT_EMA400 | -19.80 |
| SELL | 2024-08-28 15:15:00 | 628.30 | 2024-08-29 11:15:00 | 649.00 | EXIT_EMA400 | -20.70 |
| SELL | 2025-02-14 09:15:00 | 545.05 | 2025-03-25 10:15:00 | 557.05 | EXIT_EMA400 | -12.00 |
| BUY | 2026-01-16 09:15:00 | 1006.65 | 2026-01-27 11:15:00 | 950.05 | EXIT_EMA400 | -56.60 |
