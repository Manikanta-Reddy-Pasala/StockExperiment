# Anthem Biosciences Ltd. (ANTHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-07-21 09:15:00 → 2026-04-30 15:15:00 (1328 bars)
- **Last close:** 760.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -38.20
- **Avg P&L per closed trade:** -12.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 728.60 | 784.52 | 784.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 723.65 | 782.02 | 783.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 685.40 | 670.24 | 700.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 12:15:00 | 666.15 | 673.35 | 698.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 12:15:00 | 648.90 | 621.32 | 645.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 700.00 | 660.85 | 660.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 707.75 | 661.31 | 660.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 668.30 | 669.66 | 665.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 09:15:00 | 677.85 | 667.61 | 664.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-11 09:15:00 | 645.50 | 667.60 | 665.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 637.55 | 662.71 | 662.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 629.25 | 657.58 | 660.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 656.60 | 653.52 | 657.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 641.30 | 653.02 | 657.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-27 13:15:00 | 664.40 | 652.86 | 657.04 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 11:15:00 | 728.60 | 660.57 | 660.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 741.75 | 666.22 | 663.21 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-12-17 12:15:00 | 666.15 | 2026-02-06 12:15:00 | 648.90 | EXIT_EMA400 | 17.25 |
| BUY | 2026-03-10 09:15:00 | 677.85 | 2026-03-11 09:15:00 | 645.50 | EXIT_EMA400 | -32.35 |
| SELL | 2026-03-27 09:15:00 | 641.30 | 2026-03-27 13:15:00 | 664.40 | EXIT_EMA400 | -23.10 |
