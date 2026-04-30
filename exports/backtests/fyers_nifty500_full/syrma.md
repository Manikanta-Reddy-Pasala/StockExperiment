# Syrma SGS Technology Ltd. (SYRMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 954.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 8.07
- **Avg P&L per closed trade:** 2.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 415.50 | 466.88 | 466.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 410.50 | 465.31 | 466.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 457.40 | 447.09 | 455.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 10:15:00 | 440.00 | 447.30 | 455.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-12 13:15:00 | 455.55 | 437.66 | 446.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 14:15:00 | 504.25 | 438.33 | 438.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 519.90 | 445.13 | 441.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 564.60 | 588.96 | 556.86 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 412.50 | 539.30 | 539.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 10:15:00 | 405.30 | 537.97 | 538.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 531.80 | 530.49 | 534.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-30 11:15:00 | 509.00 | 530.14 | 534.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 523.40 | 529.81 | 534.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-01 09:15:00 | 541.90 | 529.81 | 534.22 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 538.00 | 478.72 | 478.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 545.05 | 481.17 | 479.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 523.90 | 526.51 | 511.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 512.80 | 525.66 | 513.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 733.60 | 795.88 | 795.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 728.20 | 792.81 | 794.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 753.45 | 753.04 | 769.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 744.85 | 752.56 | 767.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 769.95 | 713.24 | 737.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 882.70 | 755.38 | 755.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 890.60 | 775.75 | 765.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 09:15:00 | 743.25 | 780.05 | 780.12 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 816.00 | 779.89 | 779.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 837.55 | 787.18 | 783.79 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-26 10:15:00 | 440.00 | 2024-09-12 13:15:00 | 455.55 | EXIT_EMA400 | -15.55 |
| SELL | 2025-01-30 11:15:00 | 509.00 | 2025-02-01 09:15:00 | 541.90 | EXIT_EMA400 | -32.90 |
| BUY | 2025-06-13 10:15:00 | 523.90 | 2025-06-19 12:15:00 | 512.80 | EXIT_EMA400 | -11.10 |
| SELL | 2026-01-08 09:15:00 | 744.85 | 2026-01-20 09:15:00 | 677.23 | TARGET | 67.62 |
