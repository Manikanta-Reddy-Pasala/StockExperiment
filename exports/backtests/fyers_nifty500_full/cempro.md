# Cemindia Projects Ltd. (CEMPRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 815.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 69.18
- **Avg P&L per closed trade:** 11.53

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 485.00 | 539.74 | 539.77 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 11:15:00 | 538.05 | 526.15 | 526.11 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 522.45 | 526.25 | 526.27 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 12:15:00 | 530.30 | 526.31 | 526.29 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 525.10 | 526.27 | 526.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 524.50 | 526.25 | 526.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 534.90 | 526.22 | 526.25 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 14:15:00 | 537.95 | 526.34 | 526.31 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 12:15:00 | 523.15 | 526.26 | 526.27 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 13:15:00 | 534.75 | 526.24 | 526.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 15:15:00 | 537.95 | 526.44 | 526.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 543.05 | 547.06 | 539.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 557.50 | 546.93 | 539.71 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-08 09:15:00 | 539.00 | 546.85 | 539.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 497.90 | 535.77 | 535.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 494.80 | 534.35 | 535.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 14:15:00 | 531.70 | 527.67 | 531.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-06 09:15:00 | 507.55 | 527.51 | 531.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 507.55 | 527.51 | 531.42 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-06 12:15:00 | 503.70 | 526.92 | 531.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-12 09:15:00 | 546.30 | 525.23 | 529.65 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 631.00 | 534.45 | 534.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 634.00 | 536.33 | 534.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 831.40 | 832.68 | 763.45 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 721.00 | 760.16 | 760.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 713.95 | 757.58 | 759.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 756.10 | 751.60 | 755.72 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 782.10 | 759.39 | 759.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 788.95 | 759.92 | 759.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 14:15:00 | 774.60 | 774.69 | 767.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-23 09:15:00 | 821.10 | 776.50 | 768.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 779.50 | 785.26 | 774.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-30 09:15:00 | 794.05 | 785.37 | 775.02 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 09:15:00 | 783.90 | 799.83 | 786.29 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 784.45 | 807.04 | 807.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 776.30 | 806.74 | 806.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 696.50 | 692.57 | 731.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 681.55 | 692.62 | 731.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 604.25 | 560.63 | 602.38 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 804.25 | 624.56 | 623.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 815.25 | 626.46 | 624.79 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-07 15:15:00 | 557.50 | 2025-04-08 09:15:00 | 539.00 | EXIT_EMA400 | -18.50 |
| SELL | 2025-05-06 09:15:00 | 507.55 | 2025-05-12 09:15:00 | 546.30 | EXIT_EMA400 | -38.75 |
| SELL | 2025-05-06 12:15:00 | 503.70 | 2025-05-12 09:15:00 | 546.30 | EXIT_EMA400 | -42.60 |
| BUY | 2025-09-30 09:15:00 | 794.05 | 2025-10-06 09:15:00 | 851.15 | TARGET | 57.10 |
| BUY | 2025-09-23 09:15:00 | 821.10 | 2025-10-14 09:15:00 | 783.90 | EXIT_EMA400 | -37.20 |
| SELL | 2026-02-05 09:15:00 | 681.55 | 2026-03-16 09:15:00 | 532.42 | TARGET | 149.13 |
