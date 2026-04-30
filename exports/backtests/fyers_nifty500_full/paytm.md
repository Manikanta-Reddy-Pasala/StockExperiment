# One 97 Communications Ltd. (PAYTM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1096.00
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
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 70.17
- **Avg P&L per closed trade:** 17.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 761.00 | 858.27 | 858.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 741.75 | 856.00 | 857.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 743.55 | 735.74 | 772.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 721.15 | 737.45 | 771.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 12:15:00 | 777.45 | 738.98 | 769.71 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 866.05 | 785.94 | 785.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 870.50 | 792.61 | 789.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 830.10 | 830.84 | 812.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 09:15:00 | 861.65 | 831.24 | 812.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 816.50 | 832.62 | 814.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-07 09:15:00 | 861.65 | 832.91 | 814.84 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 825.55 | 843.34 | 827.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 1171.80 | 1279.94 | 1280.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1160.00 | 1272.57 | 1276.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1201.50 | 1200.51 | 1229.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 09:15:00 | 1163.20 | 1200.13 | 1229.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 1104.45 | 1052.33 | 1105.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 14:15:00 | 1112.00 | 1053.41 | 1105.57 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-20 09:15:00 | 721.15 | 2025-03-24 12:15:00 | 777.45 | EXIT_EMA400 | -56.30 |
| BUY | 2025-05-05 09:15:00 | 861.65 | 2025-05-22 12:15:00 | 825.55 | EXIT_EMA400 | -36.10 |
| BUY | 2025-05-07 09:15:00 | 861.65 | 2025-05-22 12:15:00 | 825.55 | EXIT_EMA400 | -36.10 |
| SELL | 2026-02-19 09:15:00 | 1163.20 | 2026-03-30 14:15:00 | 964.53 | TARGET | 198.67 |
