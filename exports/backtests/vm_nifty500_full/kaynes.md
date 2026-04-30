# Kaynes Technology India Ltd. (KAYNES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4044.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 1124.55
- **Avg P&L per closed trade:** 187.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 2460.00 | 2745.82 | 2747.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 2458.60 | 2706.59 | 2726.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 2668.00 | 2666.41 | 2700.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-24 14:15:00 | 2629.75 | 2666.08 | 2699.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-26 13:15:00 | 2714.10 | 2664.63 | 2696.83 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 10:15:00 | 3440.15 | 2687.79 | 2687.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 11:15:00 | 3459.00 | 2695.46 | 2691.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 2978.10 | 2999.50 | 2868.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-07 14:15:00 | 3337.85 | 3016.38 | 2891.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 4844.50 | 5170.11 | 4831.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-08 11:15:00 | 5074.35 | 5149.98 | 4834.10 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-13 09:15:00 | 6386.70 | 6863.71 | 6423.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 14:15:00 | 4943.05 | 6196.94 | 6201.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 10:15:00 | 4858.25 | 6159.29 | 6182.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 12:15:00 | 4468.05 | 4463.82 | 4895.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 4234.50 | 4670.89 | 4865.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 4823.00 | 4655.29 | 4846.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-09 15:15:00 | 4848.00 | 4664.14 | 4843.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 13:15:00 | 5972.90 | 4984.17 | 4983.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6052.00 | 5474.00 | 5285.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 14:15:00 | 5821.00 | 5842.90 | 5598.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 5900.50 | 5723.49 | 5623.70 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 5773.00 | 5946.13 | 5819.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 6239.00 | 6673.99 | 6675.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 6210.00 | 6661.03 | 6669.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3836.00 | 3771.76 | 4299.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 10:15:00 | 3761.50 | 3867.84 | 4146.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3835.40 | 3667.80 | 3858.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 10:15:00 | 3871.50 | 3669.83 | 3858.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 4137.90 | 3964.10 | 3963.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4166.00 | 3971.76 | 3967.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 12:15:00 | 3977.70 | 3980.26 | 3972.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 13:15:00 | 4020.00 | 3980.66 | 3972.44 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-24 14:15:00 | 2629.75 | 2024-04-26 13:15:00 | 2714.10 | EXIT_EMA400 | -84.35 |
| BUY | 2024-06-07 14:15:00 | 3337.85 | 2024-07-29 11:15:00 | 4677.02 | TARGET | 1339.17 |
| BUY | 2024-10-08 11:15:00 | 5074.35 | 2024-10-15 09:15:00 | 5795.09 | TARGET | 720.74 |
| SELL | 2025-04-07 09:15:00 | 4234.50 | 2025-04-09 15:15:00 | 4848.00 | EXIT_EMA400 | -613.50 |
| BUY | 2025-06-24 10:15:00 | 5900.50 | 2025-07-21 09:15:00 | 5773.00 | EXIT_EMA400 | -127.50 |
| SELL | 2026-03-02 10:15:00 | 3761.50 | 2026-04-08 10:15:00 | 3871.50 | EXIT_EMA400 | -110.00 |
