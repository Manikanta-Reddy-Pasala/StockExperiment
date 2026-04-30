# Motilal Oswal Financial Services Ltd. (MOTILALOFS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 800.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
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
- **Total realized P&L (per unit):** 1.32
- **Avg P&L per closed trade:** 0.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 740.55 | 896.95 | 897.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 721.40 | 892.26 | 894.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 635.80 | 626.46 | 689.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 15:15:00 | 611.00 | 628.75 | 679.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 651.85 | 616.28 | 653.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 09:15:00 | 692.50 | 617.39 | 653.23 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 738.00 | 672.46 | 672.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 10:15:00 | 741.85 | 673.79 | 673.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 894.50 | 895.72 | 849.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 11:15:00 | 903.65 | 895.48 | 850.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 891.90 | 919.19 | 889.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-28 15:15:00 | 887.75 | 918.88 | 889.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 844.85 | 953.18 | 953.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 826.40 | 950.88 | 952.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 870.40 | 863.56 | 890.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 13:15:00 | 856.90 | 863.52 | 889.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 745.00 | 700.77 | 741.28 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 15:15:00 | 611.00 | 2025-04-21 09:15:00 | 692.50 | EXIT_EMA400 | -81.50 |
| BUY | 2025-07-29 11:15:00 | 903.65 | 2025-08-28 15:15:00 | 887.75 | EXIT_EMA400 | -15.90 |
| SELL | 2026-01-16 13:15:00 | 856.90 | 2026-01-23 13:15:00 | 758.18 | TARGET | 98.72 |
