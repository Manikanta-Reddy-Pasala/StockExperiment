# JSW Energy Ltd. (JSWENERGY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 561.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -43.01
- **Avg P&L per closed trade:** -7.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 12:15:00 | 663.00 | 710.66 | 710.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 646.20 | 706.96 | 708.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 718.10 | 691.09 | 699.20 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 732.00 | 705.76 | 705.70 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 694.75 | 705.71 | 705.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 14:15:00 | 689.70 | 705.22 | 705.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 705.45 | 705.01 | 705.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 693.65 | 704.83 | 705.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 674.30 | 704.44 | 705.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-25 15:15:00 | 668.80 | 704.08 | 704.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-16 09:15:00 | 693.20 | 680.71 | 689.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 522.00 | 507.89 | 507.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 525.30 | 508.35 | 508.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 509.30 | 509.56 | 508.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-07 09:15:00 | 520.50 | 509.58 | 508.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 509.10 | 509.67 | 508.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-07 14:15:00 | 508.80 | 509.67 | 508.86 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 505.50 | 518.57 | 518.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 504.45 | 518.29 | 518.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 517.70 | 516.08 | 517.28 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 532.00 | 518.35 | 518.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 545.50 | 518.62 | 518.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 527.45 | 528.08 | 523.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 15:15:00 | 537.00 | 528.15 | 524.45 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-20 09:15:00 | 522.55 | 535.77 | 530.13 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 506.75 | 528.44 | 528.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 505.00 | 528.01 | 528.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 12:15:00 | 484.70 | 484.57 | 498.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 09:15:00 | 482.50 | 484.56 | 498.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 495.30 | 482.70 | 494.95 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 513.60 | 486.69 | 486.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 517.65 | 491.96 | 489.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 489.05 | 493.03 | 490.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-25 10:15:00 | 502.50 | 492.26 | 489.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 491.60 | 492.46 | 490.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-27 10:15:00 | 488.00 | 492.41 | 490.05 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 12:15:00 | 693.65 | 2024-11-26 14:15:00 | 658.81 | TARGET | 34.84 |
| SELL | 2024-11-25 15:15:00 | 668.80 | 2024-12-16 09:15:00 | 693.20 | EXIT_EMA400 | -24.40 |
| BUY | 2025-07-07 09:15:00 | 520.50 | 2025-07-07 14:15:00 | 508.80 | EXIT_EMA400 | -11.70 |
| BUY | 2025-10-01 15:15:00 | 537.00 | 2025-10-20 09:15:00 | 522.55 | EXIT_EMA400 | -14.45 |
| SELL | 2025-12-24 09:15:00 | 482.50 | 2026-01-01 09:15:00 | 495.30 | EXIT_EMA400 | -12.80 |
| BUY | 2026-03-25 10:15:00 | 502.50 | 2026-03-27 10:15:00 | 488.00 | EXIT_EMA400 | -14.50 |
