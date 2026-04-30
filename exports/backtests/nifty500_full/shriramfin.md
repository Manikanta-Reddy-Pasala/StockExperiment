# Shriram Finance Ltd. (SHRIRAMFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 937.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
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
- **Total realized P&L (per unit):** -153.94
- **Avg P&L per closed trade:** -38.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 572.28 | 638.94 | 639.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 569.94 | 637.01 | 638.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 617.40 | 611.49 | 622.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 12:15:00 | 601.62 | 621.86 | 624.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-02 14:15:00 | 611.91 | 601.49 | 611.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 631.60 | 576.35 | 576.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 642.90 | 577.56 | 576.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 606.35 | 635.14 | 615.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 644.40 | 633.59 | 615.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-28 09:15:00 | 606.90 | 656.83 | 634.26 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 637.50 | 658.16 | 658.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 631.25 | 657.32 | 657.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 610.80 | 609.78 | 624.69 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 666.90 | 629.51 | 629.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 669.20 | 630.26 | 629.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.60 | 972.06 | 913.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 1004.60 | 971.30 | 914.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 954.50 | 1035.12 | 989.85 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 13:15:00 | 891.60 | 973.65 | 973.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 1025.80 | 973.92 | 973.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 14:15:00 | 1028.40 | 974.46 | 973.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 997.04 | 986.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-24 14:15:00 | 1021.80 | 997.36 | 987.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.80 | 997.36 | 987.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-27 09:15:00 | 965.75 | 997.23 | 987.28 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 12:15:00 | 601.62 | 2025-01-02 14:15:00 | 611.91 | EXIT_EMA400 | -10.29 |
| BUY | 2025-04-08 09:15:00 | 644.40 | 2025-04-28 09:15:00 | 606.90 | EXIT_EMA400 | -37.50 |
| BUY | 2026-02-03 09:15:00 | 1004.60 | 2026-03-09 09:15:00 | 954.50 | EXIT_EMA400 | -50.10 |
| BUY | 2026-04-24 14:15:00 | 1021.80 | 2026-04-27 09:15:00 | 965.75 | EXIT_EMA400 | -56.05 |
