# Indian Bank (INDIANB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 851.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -75.87
- **Avg P&L per closed trade:** -15.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 519.65 | 553.04 | 553.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 518.05 | 551.21 | 552.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 535.10 | 534.43 | 542.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 10:15:00 | 521.60 | 533.89 | 541.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 539.40 | 533.09 | 540.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-30 09:15:00 | 531.60 | 533.10 | 540.29 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-10-28 13:15:00 | 552.50 | 519.40 | 528.39 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 582.30 | 536.10 | 536.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 592.65 | 553.31 | 546.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 561.60 | 567.88 | 556.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 09:15:00 | 584.95 | 568.16 | 557.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 559.15 | 568.74 | 558.30 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 557.80 | 568.63 | 558.30 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 527.65 | 551.77 | 551.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 523.50 | 550.26 | 551.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 529.20 | 528.17 | 537.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 518.70 | 528.18 | 537.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 13:15:00 | 549.50 | 523.16 | 532.70 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 530.20 | 527.05 | 527.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 543.95 | 527.30 | 527.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 526.90 | 529.77 | 528.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 14:15:00 | 540.40 | 529.95 | 528.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-09 10:15:00 | 527.30 | 531.06 | 529.23 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 09:15:00 | 531.60 | 2024-10-22 10:15:00 | 505.52 | TARGET | 26.08 |
| SELL | 2024-09-25 10:15:00 | 521.60 | 2024-10-28 13:15:00 | 552.50 | EXIT_EMA400 | -30.90 |
| BUY | 2024-12-16 09:15:00 | 584.95 | 2024-12-18 13:15:00 | 557.80 | EXIT_EMA400 | -27.15 |
| SELL | 2025-01-22 09:15:00 | 518.70 | 2025-01-29 13:15:00 | 549.50 | EXIT_EMA400 | -30.80 |
| BUY | 2025-04-07 14:15:00 | 540.40 | 2025-04-09 10:15:00 | 527.30 | EXIT_EMA400 | -13.10 |
