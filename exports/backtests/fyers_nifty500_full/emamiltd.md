# Emami Ltd. (EMAMILTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 445.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -148.75
- **Avg P&L per closed trade:** -29.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 721.90 | 760.88 | 760.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 715.60 | 755.39 | 758.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 716.45 | 709.55 | 730.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 674.70 | 709.29 | 729.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 15:15:00 | 732.00 | 675.98 | 702.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 618.85 | 575.76 | 575.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 625.80 | 576.25 | 575.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 618.70 | 618.82 | 605.00 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 583.00 | 597.95 | 598.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 581.05 | 596.38 | 597.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 579.95 | 579.73 | 586.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 11:15:00 | 560.65 | 579.54 | 586.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 578.80 | 575.50 | 583.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-07 13:15:00 | 575.80 | 575.54 | 583.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-09 10:15:00 | 593.00 | 575.64 | 583.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 11:15:00 | 596.45 | 585.66 | 585.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 601.25 | 585.93 | 585.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 574.50 | 585.60 | 585.61 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 604.00 | 585.54 | 585.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 608.40 | 585.91 | 585.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 569.25 | 592.02 | 588.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-04 09:15:00 | 612.75 | 588.42 | 587.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 589.40 | 591.55 | 589.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-09 14:15:00 | 587.65 | 591.48 | 589.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 557.15 | 588.90 | 589.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 555.35 | 588.27 | 588.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 528.55 | 528.43 | 543.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 12:15:00 | 523.35 | 528.21 | 541.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 535.25 | 526.12 | 537.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-12 14:15:00 | 540.15 | 526.48 | 537.41 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-04 09:15:00 | 674.70 | 2024-11-25 15:15:00 | 732.00 | EXIT_EMA400 | -57.30 |
| SELL | 2025-07-01 11:15:00 | 560.65 | 2025-07-09 10:15:00 | 593.00 | EXIT_EMA400 | -32.35 |
| SELL | 2025-07-07 13:15:00 | 575.80 | 2025-07-09 10:15:00 | 593.00 | EXIT_EMA400 | -17.20 |
| BUY | 2025-09-04 09:15:00 | 612.75 | 2025-09-09 14:15:00 | 587.65 | EXIT_EMA400 | -25.10 |
| SELL | 2025-12-02 12:15:00 | 523.35 | 2025-12-12 14:15:00 | 540.15 | EXIT_EMA400 | -16.80 |
