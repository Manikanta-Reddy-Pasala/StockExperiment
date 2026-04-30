# Authum Investment & Infrastructure Ltd. (AIIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 477.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** 10.04
- **Avg P&L per closed trade:** 2.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 11:15:00 | 301.12 | 344.61 | 344.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 12:15:00 | 297.60 | 344.14 | 344.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 09:15:00 | 313.55 | 308.74 | 321.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 14:15:00 | 307.01 | 310.56 | 320.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 315.99 | 310.58 | 320.06 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-27 13:15:00 | 321.74 | 310.83 | 319.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 11:15:00 | 364.06 | 326.26 | 326.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 15:15:00 | 370.00 | 327.83 | 326.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 510.84 | 530.89 | 492.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 14:15:00 | 565.00 | 532.23 | 496.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 558.68 | 553.88 | 521.42 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-14 13:15:00 | 577.60 | 554.11 | 521.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-24 13:15:00 | 607.96 | 628.92 | 608.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 11:15:00 | 553.84 | 595.90 | 596.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 544.02 | 590.05 | 592.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 543.74 | 541.49 | 559.65 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 609.38 | 568.74 | 568.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 618.60 | 569.62 | 569.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 09:15:00 | 576.30 | 600.72 | 587.79 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 513.90 | 577.42 | 577.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 507.50 | 574.85 | 576.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 481.45 | 480.51 | 508.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 09:15:00 | 444.45 | 485.21 | 504.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 13:15:00 | 493.00 | 460.04 | 479.95 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 14:15:00 | 307.01 | 2025-03-27 13:15:00 | 321.74 | EXIT_EMA400 | -14.73 |
| BUY | 2025-07-29 14:15:00 | 565.00 | 2025-10-24 13:15:00 | 607.96 | EXIT_EMA400 | 42.96 |
| BUY | 2025-08-14 13:15:00 | 577.60 | 2025-10-24 13:15:00 | 607.96 | EXIT_EMA400 | 30.36 |
| SELL | 2026-03-30 09:15:00 | 444.45 | 2026-04-22 13:15:00 | 493.00 | EXIT_EMA400 | -48.55 |
