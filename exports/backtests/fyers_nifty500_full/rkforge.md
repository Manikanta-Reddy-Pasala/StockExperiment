# Ramkrishna Forgings Ltd. (RKFORGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 597.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 82.11
- **Avg P&L per closed trade:** 27.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 10:15:00 | 905.25 | 952.06 | 952.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 900.90 | 951.09 | 951.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 11:15:00 | 981.55 | 926.34 | 935.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 904.00 | 939.95 | 941.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 754.00 | 713.65 | 764.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 773.65 | 717.06 | 763.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 569.90 | 526.86 | 526.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 571.00 | 528.89 | 527.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 544.00 | 544.18 | 537.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-27 09:15:00 | 551.35 | 544.12 | 537.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 550.05 | 544.56 | 538.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-02 14:15:00 | 557.40 | 544.99 | 538.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 544.50 | 548.98 | 542.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 10:15:00 | 539.50 | 548.88 | 542.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 480.70 | 537.60 | 537.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 468.90 | 528.85 | 533.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 525.10 | 519.54 | 527.36 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 13:15:00 | 547.40 | 532.07 | 532.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 551.45 | 532.53 | 532.24 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-22 09:15:00 | 904.00 | 2025-01-27 09:15:00 | 792.14 | TARGET | 111.86 |
| BUY | 2026-02-27 09:15:00 | 551.35 | 2026-03-13 10:15:00 | 539.50 | EXIT_EMA400 | -11.85 |
| BUY | 2026-03-02 14:15:00 | 557.40 | 2026-03-13 10:15:00 | 539.50 | EXIT_EMA400 | -17.90 |
