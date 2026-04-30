# Motilal Oswal Financial Services Ltd. (MOTILALOFS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 807.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 56.57
- **Avg P&L per closed trade:** 18.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 740.55 | 897.06 | 897.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 721.90 | 892.38 | 895.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 635.80 | 625.26 | 687.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-01 10:15:00 | 607.85 | 626.50 | 672.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 651.85 | 616.01 | 651.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 15:15:00 | 652.30 | 616.37 | 651.95 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 740.00 | 671.78 | 671.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 748.00 | 674.50 | 672.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 894.70 | 895.70 | 849.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 11:15:00 | 903.65 | 895.48 | 850.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 891.90 | 919.18 | 889.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-28 15:15:00 | 889.00 | 918.88 | 889.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 844.60 | 953.12 | 953.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 826.00 | 950.82 | 952.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 870.35 | 863.51 | 890.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 14:15:00 | 851.00 | 863.34 | 889.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 745.00 | 700.56 | 740.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 807.00 | 762.39 | 762.21 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-01 10:15:00 | 607.85 | 2025-04-17 15:15:00 | 652.30 | EXIT_EMA400 | -44.45 |
| BUY | 2025-07-29 11:15:00 | 903.65 | 2025-08-28 15:15:00 | 889.00 | EXIT_EMA400 | -14.65 |
| SELL | 2026-01-16 14:15:00 | 851.00 | 2026-01-27 10:15:00 | 735.33 | TARGET | 115.67 |
