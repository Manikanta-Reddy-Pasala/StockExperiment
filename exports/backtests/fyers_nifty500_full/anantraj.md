# Anant Raj Ltd. (ANANTRAJ.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 488.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -76.70
- **Avg P&L per closed trade:** -19.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 611.30 | 762.54 | 762.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 601.10 | 757.17 | 759.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 546.45 | 534.66 | 595.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 10:15:00 | 509.10 | 534.25 | 588.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-16 10:15:00 | 506.80 | 468.62 | 503.65 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 575.50 | 520.43 | 520.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 592.55 | 522.21 | 521.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 531.75 | 535.16 | 528.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 14:15:00 | 553.35 | 535.58 | 528.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 538.80 | 536.11 | 529.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 13:15:00 | 527.55 | 535.92 | 529.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 519.50 | 548.91 | 548.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 512.30 | 548.54 | 548.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 545.90 | 544.28 | 546.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 12:15:00 | 535.80 | 543.91 | 546.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 566.65 | 539.29 | 543.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 587.60 | 547.02 | 546.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 591.75 | 549.01 | 547.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 639.65 | 648.60 | 613.55 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 563.80 | 614.08 | 614.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 559.40 | 613.04 | 613.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 564.60 | 562.70 | 580.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 553.20 | 562.68 | 579.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 13:15:00 | 575.55 | 558.37 | 574.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 10:15:00 | 509.10 | 2025-05-16 10:15:00 | 506.80 | EXIT_EMA400 | 2.30 |
| BUY | 2025-06-16 14:15:00 | 553.35 | 2025-06-18 13:15:00 | 527.55 | EXIT_EMA400 | -25.80 |
| SELL | 2025-09-04 12:15:00 | 535.80 | 2025-09-15 09:15:00 | 566.65 | EXIT_EMA400 | -30.85 |
| SELL | 2025-12-26 13:15:00 | 553.20 | 2026-01-02 13:15:00 | 575.55 | EXIT_EMA400 | -22.35 |
