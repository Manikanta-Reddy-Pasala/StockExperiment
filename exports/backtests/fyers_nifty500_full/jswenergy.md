# JSW Energy Ltd. (JSWENERGY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 560.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -32.30
- **Avg P&L per closed trade:** -6.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 12:15:00 | 663.00 | 710.65 | 710.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 646.80 | 706.97 | 708.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 718.10 | 690.93 | 699.12 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 732.00 | 705.68 | 705.65 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 694.75 | 705.65 | 705.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 14:15:00 | 689.70 | 705.17 | 705.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 705.45 | 704.95 | 705.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 693.95 | 704.78 | 705.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 674.30 | 704.39 | 705.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-25 15:15:00 | 666.50 | 704.01 | 704.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-16 09:15:00 | 693.10 | 680.66 | 689.56 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 521.90 | 507.64 | 507.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 12:15:00 | 524.65 | 508.19 | 507.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 509.30 | 509.57 | 508.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-07 09:15:00 | 520.50 | 509.59 | 508.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 509.10 | 509.68 | 508.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-07 14:15:00 | 508.50 | 509.67 | 508.77 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 505.30 | 518.57 | 518.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 504.45 | 518.30 | 518.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 517.85 | 516.07 | 517.26 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 532.40 | 518.35 | 518.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 545.35 | 518.62 | 518.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 527.20 | 528.09 | 523.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 15:15:00 | 537.00 | 528.17 | 524.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-20 09:15:00 | 522.25 | 535.75 | 530.12 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 506.75 | 528.47 | 528.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 505.00 | 528.04 | 528.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 12:15:00 | 484.70 | 484.54 | 498.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 09:15:00 | 482.50 | 484.53 | 498.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 495.25 | 482.66 | 494.93 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 12:15:00 | 513.45 | 485.85 | 485.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 527.90 | 492.05 | 490.07 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 12:15:00 | 693.95 | 2024-11-26 14:15:00 | 660.15 | TARGET | 33.80 |
| SELL | 2024-11-25 15:15:00 | 666.50 | 2024-12-16 09:15:00 | 693.10 | EXIT_EMA400 | -26.60 |
| BUY | 2025-07-07 09:15:00 | 520.50 | 2025-07-07 14:15:00 | 508.50 | EXIT_EMA400 | -12.00 |
| BUY | 2025-10-01 15:15:00 | 537.00 | 2025-10-20 09:15:00 | 522.25 | EXIT_EMA400 | -14.75 |
| SELL | 2025-12-24 09:15:00 | 482.50 | 2026-01-01 09:15:00 | 495.25 | EXIT_EMA400 | -12.75 |
