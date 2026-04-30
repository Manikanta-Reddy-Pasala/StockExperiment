# Hexaware Technologies Ltd. (HEXT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-02-19 09:15:00 → 2026-04-30 15:15:00 (2046 bars)
- **Last close:** 448.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -66.86
- **Avg P&L per closed trade:** -13.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 799.40 | 726.62 | 726.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 12:15:00 | 806.50 | 728.15 | 727.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 843.25 | 843.53 | 813.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-17 10:15:00 | 867.10 | 845.33 | 817.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 826.80 | 848.84 | 824.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 773.00 | 848.08 | 824.25 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 694.00 | 804.99 | 805.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 691.35 | 802.80 | 803.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 762.30 | 759.49 | 777.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-03 09:15:00 | 745.50 | 768.98 | 778.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 772.90 | 759.82 | 772.05 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 765.85 | 726.81 | 726.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 773.00 | 735.45 | 731.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 742.20 | 744.45 | 737.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-24 10:15:00 | 750.00 | 743.95 | 737.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 738.55 | 744.11 | 737.53 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-26 14:15:00 | 751.20 | 744.18 | 737.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-29 15:15:00 | 737.50 | 744.14 | 737.96 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 708.00 | 736.21 | 736.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 691.20 | 730.13 | 733.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 722.85 | 721.12 | 727.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 700.50 | 721.10 | 727.45 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-17 10:15:00 | 867.10 | 2025-07-25 09:15:00 | 773.00 | EXIT_EMA400 | -94.10 |
| SELL | 2025-09-03 09:15:00 | 745.50 | 2025-09-10 09:15:00 | 772.90 | EXIT_EMA400 | -27.40 |
| BUY | 2025-12-24 10:15:00 | 750.00 | 2025-12-29 15:15:00 | 737.50 | EXIT_EMA400 | -12.50 |
| BUY | 2025-12-26 14:15:00 | 751.20 | 2025-12-29 15:15:00 | 737.50 | EXIT_EMA400 | -13.70 |
| SELL | 2026-02-04 09:15:00 | 700.50 | 2026-02-06 09:15:00 | 619.66 | TARGET | 80.84 |
