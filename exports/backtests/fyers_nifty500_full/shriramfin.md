# Shriram Finance Ltd. (SHRIRAMFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 938.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -153.03
- **Avg P&L per closed trade:** -38.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 572.28 | 638.60 | 638.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 569.88 | 636.68 | 637.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 617.40 | 611.32 | 622.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 12:15:00 | 601.62 | 621.77 | 624.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-02 14:15:00 | 611.80 | 601.46 | 611.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 631.55 | 575.50 | 575.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 642.90 | 577.27 | 576.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 644.40 | 633.50 | 614.70 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-28 09:15:00 | 606.90 | 656.81 | 634.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 637.50 | 658.16 | 658.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 631.25 | 657.32 | 657.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 610.80 | 609.79 | 624.69 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 666.20 | 629.15 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 670.55 | 630.67 | 629.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.50 | 974.04 | 916.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 1004.60 | 973.18 | 917.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 954.60 | 1035.44 | 991.13 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.58 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1027.50 | 974.98 | 974.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1029.25 | 978.48 | 976.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 998.77 | 988.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-27 09:15:00 | 965.75 | 998.85 | 988.75 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 12:15:00 | 601.62 | 2025-01-02 14:15:00 | 611.80 | EXIT_EMA400 | -10.18 |
| BUY | 2025-04-08 09:15:00 | 644.40 | 2025-04-28 09:15:00 | 606.90 | EXIT_EMA400 | -37.50 |
| BUY | 2026-02-03 09:15:00 | 1004.60 | 2026-03-09 09:15:00 | 954.60 | EXIT_EMA400 | -50.00 |
| BUY | 2026-04-24 14:15:00 | 1021.10 | 2026-04-27 09:15:00 | 965.75 | EXIT_EMA400 | -55.35 |
