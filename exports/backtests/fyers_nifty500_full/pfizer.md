# Pfizer Ltd. (PFIZER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4711.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 93.67
- **Avg P&L per closed trade:** 18.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 5185.00 | 5588.33 | 5588.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 10:15:00 | 5152.60 | 5584.00 | 5586.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 5344.95 | 5332.64 | 5419.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 12:15:00 | 5288.65 | 5333.21 | 5404.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 5214.85 | 5086.12 | 5223.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-30 14:15:00 | 5333.40 | 5088.58 | 5223.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 13:15:00 | 4955.00 | 4296.42 | 4296.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 5046.20 | 4364.29 | 4331.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 5591.00 | 5591.60 | 5289.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-18 11:15:00 | 5692.00 | 5262.98 | 5248.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 11:15:00 | 5316.00 | 5387.94 | 5321.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 5137.00 | 5275.30 | 5275.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 5126.00 | 5263.94 | 5269.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 5149.00 | 5141.89 | 5196.18 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 5351.00 | 5227.94 | 5227.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 13:15:00 | 5372.50 | 5233.59 | 5230.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 5252.00 | 5258.43 | 5244.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 14:15:00 | 5325.50 | 5259.11 | 5246.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 5325.50 | 5259.11 | 5246.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-31 11:15:00 | 5237.00 | 5259.18 | 5246.54 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 5070.50 | 5234.84 | 5235.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 5064.00 | 5233.14 | 5234.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 5037.00 | 5036.89 | 5092.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 13:15:00 | 4977.00 | 5046.76 | 5085.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 5117.90 | 4741.91 | 4850.98 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 5147.60 | 4923.19 | 4923.05 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4737.50 | 4924.10 | 4924.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 4711.50 | 4898.40 | 4911.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 4842.50 | 4841.09 | 4877.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 4724.50 | 4839.46 | 4876.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-01 12:15:00 | 4838.70 | 4787.28 | 4838.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-05 12:15:00 | 5288.65 | 2024-12-16 09:15:00 | 4941.70 | TARGET | 346.95 |
| BUY | 2025-08-18 11:15:00 | 5692.00 | 2025-08-26 11:15:00 | 5316.00 | EXIT_EMA400 | -376.00 |
| BUY | 2025-10-30 14:15:00 | 5325.50 | 2025-10-31 11:15:00 | 5237.00 | EXIT_EMA400 | -88.50 |
| SELL | 2025-12-29 13:15:00 | 4977.00 | 2026-01-20 09:15:00 | 4651.58 | TARGET | 325.42 |
| SELL | 2026-03-19 09:15:00 | 4724.50 | 2026-04-01 12:15:00 | 4838.70 | EXIT_EMA400 | -114.20 |
