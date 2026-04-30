# Indian Bank (INDIANB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 849.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -104.20
- **Avg P&L per closed trade:** -17.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 521.85 | 551.83 | 551.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 518.10 | 551.20 | 551.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 535.05 | 534.42 | 541.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 11:15:00 | 520.10 | 533.73 | 540.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 539.40 | 533.08 | 540.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-30 09:15:00 | 531.55 | 533.09 | 539.95 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-10-28 13:15:00 | 552.20 | 519.38 | 528.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 572.00 | 535.75 | 535.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 577.85 | 536.16 | 535.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 546.20 | 547.45 | 542.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-25 09:15:00 | 570.05 | 542.35 | 540.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 570.05 | 542.35 | 540.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-04 12:15:00 | 598.95 | 554.73 | 547.75 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 559.15 | 568.88 | 558.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 557.80 | 568.77 | 558.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 527.65 | 551.84 | 551.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 512.10 | 549.94 | 550.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 529.20 | 528.20 | 537.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 518.70 | 528.20 | 537.09 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 13:15:00 | 549.50 | 523.18 | 532.75 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 529.75 | 527.15 | 527.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 543.95 | 527.32 | 527.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 526.60 | 529.78 | 528.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 14:15:00 | 540.40 | 529.97 | 528.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-09 10:15:00 | 527.30 | 531.08 | 529.28 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 09:15:00 | 531.55 | 2024-10-22 10:15:00 | 506.35 | TARGET | 25.20 |
| SELL | 2024-09-25 11:15:00 | 520.10 | 2024-10-28 13:15:00 | 552.20 | EXIT_EMA400 | -32.10 |
| BUY | 2024-11-25 09:15:00 | 570.05 | 2024-12-18 13:15:00 | 557.80 | EXIT_EMA400 | -12.25 |
| BUY | 2024-12-04 12:15:00 | 598.95 | 2024-12-18 13:15:00 | 557.80 | EXIT_EMA400 | -41.15 |
| SELL | 2025-01-22 09:15:00 | 518.70 | 2025-01-29 13:15:00 | 549.50 | EXIT_EMA400 | -30.80 |
| BUY | 2025-04-07 14:15:00 | 540.40 | 2025-04-09 10:15:00 | 527.30 | EXIT_EMA400 | -13.10 |
