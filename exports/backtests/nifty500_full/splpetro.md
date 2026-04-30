# Supreme Petrochem Ltd. (SPLPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 757.30
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
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 14.46
- **Avg P&L per closed trade:** 3.62

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 778.85 | 835.98 | 836.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 775.55 | 832.20 | 834.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 743.00 | 735.33 | 764.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-16 11:15:00 | 728.40 | 749.84 | 765.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 667.60 | 638.67 | 674.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-06 13:15:00 | 677.85 | 641.55 | 674.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 652.60 | 624.81 | 624.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 655.75 | 627.43 | 626.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 804.45 | 804.89 | 767.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-23 09:15:00 | 817.75 | 805.19 | 769.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 772.80 | 806.26 | 781.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 754.85 | 796.27 | 796.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 749.10 | 794.98 | 795.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 679.45 | 669.48 | 707.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 14:15:00 | 603.15 | 650.40 | 676.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 630.80 | 576.08 | 616.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 654.50 | 631.82 | 631.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 664.75 | 632.14 | 631.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 652.65 | 652.88 | 643.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 11:15:00 | 666.90 | 653.53 | 644.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 668.45 | 655.04 | 646.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 643.25 | 655.30 | 646.47 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-16 11:15:00 | 728.40 | 2025-01-22 14:15:00 | 617.69 | TARGET | 110.71 |
| BUY | 2025-07-23 09:15:00 | 817.75 | 2025-08-07 10:15:00 | 772.80 | EXIT_EMA400 | -44.95 |
| SELL | 2026-01-06 14:15:00 | 603.15 | 2026-02-03 11:15:00 | 630.80 | EXIT_EMA400 | -27.65 |
| BUY | 2026-03-17 11:15:00 | 666.90 | 2026-03-23 09:15:00 | 643.25 | EXIT_EMA400 | -23.65 |
