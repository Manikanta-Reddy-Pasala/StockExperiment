# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 541.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -25.65
- **Avg P&L per closed trade:** -12.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 765.35 | 744.59 | 744.54 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 738.45 | 744.48 | 744.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 732.50 | 744.37 | 744.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 749.95 | 742.42 | 743.44 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 13:15:00 | 764.25 | 744.47 | 744.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 773.05 | 745.13 | 744.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 764.70 | 769.22 | 759.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 10:15:00 | 781.95 | 769.11 | 760.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 770.80 | 774.02 | 765.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 762.00 | 773.72 | 765.24 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 729.15 | 766.97 | 767.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 724.75 | 763.29 | 765.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 727.95 | 722.97 | 734.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 721.75 | 725.65 | 733.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-20 09:15:00 | 727.45 | 715.43 | 723.26 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-05 10:15:00 | 781.95 | 2025-06-16 09:15:00 | 762.00 | EXIT_EMA400 | -19.95 |
| SELL | 2025-09-23 10:15:00 | 721.75 | 2025-10-20 09:15:00 | 727.45 | EXIT_EMA400 | -5.70 |
