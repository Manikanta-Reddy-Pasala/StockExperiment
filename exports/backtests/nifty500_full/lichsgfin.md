# LIC Housing Finance Ltd. (LICHSGFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 554.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -66.10
- **Avg P&L per closed trade:** -13.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 649.30 | 714.56 | 714.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 641.50 | 679.77 | 689.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 634.95 | 631.36 | 653.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 11:15:00 | 628.15 | 632.00 | 652.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 633.85 | 625.12 | 639.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-29 11:15:00 | 632.35 | 625.82 | 638.93 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 638.85 | 626.13 | 638.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 09:15:00 | 624.95 | 626.25 | 638.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 639.55 | 626.42 | 638.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 10:15:00 | 615.40 | 562.31 | 562.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 620.00 | 577.46 | 570.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 584.40 | 584.85 | 575.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 592.75 | 584.93 | 575.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 578.00 | 585.70 | 576.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 15:15:00 | 574.50 | 585.59 | 576.72 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 574.40 | 600.88 | 600.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 573.50 | 600.61 | 600.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 573.30 | 569.88 | 579.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 13:15:00 | 564.30 | 574.60 | 579.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 577.20 | 573.44 | 578.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-03 09:15:00 | 578.95 | 573.49 | 578.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 558.50 | 520.49 | 520.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 561.65 | 521.29 | 520.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-31 11:15:00 | 628.15 | 2024-12-03 09:15:00 | 639.55 | EXIT_EMA400 | -11.40 |
| SELL | 2024-11-29 11:15:00 | 632.35 | 2024-12-03 09:15:00 | 639.55 | EXIT_EMA400 | -7.20 |
| SELL | 2024-12-02 09:15:00 | 624.95 | 2024-12-03 09:15:00 | 639.55 | EXIT_EMA400 | -14.60 |
| BUY | 2025-05-07 10:15:00 | 592.75 | 2025-05-08 15:15:00 | 574.50 | EXIT_EMA400 | -18.25 |
| SELL | 2025-09-26 13:15:00 | 564.30 | 2025-10-03 09:15:00 | 578.95 | EXIT_EMA400 | -14.65 |
