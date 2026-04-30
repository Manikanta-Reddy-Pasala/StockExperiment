# PI Industries Ltd. (PIIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3080.00
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
- **Total realized P&L (per unit):** 82.20
- **Avg P&L per closed trade:** 27.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 4140.85 | 4426.45 | 4427.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 10:15:00 | 4095.70 | 4423.16 | 4425.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 3689.35 | 3593.03 | 3782.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 3416.80 | 3586.80 | 3743.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 3418.40 | 3281.24 | 3440.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 10:15:00 | 3453.00 | 3306.40 | 3436.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 3636.00 | 3472.85 | 3472.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 3659.20 | 3474.70 | 3473.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 3620.00 | 3622.94 | 3569.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-22 13:15:00 | 3650.10 | 3623.21 | 3569.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4045.00 | 4116.97 | 4024.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 11:15:00 | 4020.00 | 4115.15 | 4024.53 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 3865.30 | 3963.08 | 3963.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3826.90 | 3956.79 | 3960.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3612.40 | 3610.69 | 3696.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 12:15:00 | 3558.20 | 3606.88 | 3681.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-03 09:15:00 | 3680.40 | 3606.59 | 3680.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 09:15:00 | 3416.80 | 2025-03-21 10:15:00 | 3453.00 | EXIT_EMA400 | -36.20 |
| BUY | 2025-05-22 13:15:00 | 3650.10 | 2025-05-29 09:15:00 | 3890.70 | TARGET | 240.60 |
| SELL | 2025-10-31 12:15:00 | 3558.20 | 2025-11-03 09:15:00 | 3680.40 | EXIT_EMA400 | -122.20 |
