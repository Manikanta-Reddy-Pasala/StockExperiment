# Poonawalla Fincorp Ltd. (POONAWALLA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 418.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -57.00
- **Avg P&L per closed trade:** -14.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 12:15:00 | 350.00 | 313.67 | 313.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 353.50 | 317.31 | 315.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 446.00 | 446.64 | 425.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 450.10 | 437.73 | 428.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 14:15:00 | 427.90 | 448.87 | 437.52 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 461.90 | 474.79 | 474.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 460.20 | 474.64 | 474.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 463.95 | 460.50 | 466.47 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 475.00 | 470.22 | 470.22 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 463.00 | 470.16 | 470.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 460.75 | 470.06 | 470.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 13:15:00 | 468.85 | 467.10 | 468.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 12:15:00 | 463.95 | 467.17 | 468.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-19 09:15:00 | 478.00 | 467.20 | 468.52 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 463.90 | 456.03 | 456.03 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 436.75 | 455.95 | 456.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 431.45 | 455.20 | 455.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 425.60 | 409.40 | 425.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 10:15:00 | 410.35 | 410.05 | 425.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 420.00 | 408.95 | 421.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-20 13:15:00 | 418.50 | 409.04 | 421.50 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 424.80 | 409.38 | 421.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-08-13 09:15:00 | 450.10 | 2025-08-28 14:15:00 | 427.90 | EXIT_EMA400 | -22.20 |
| SELL | 2026-01-16 12:15:00 | 463.95 | 2026-01-19 09:15:00 | 478.00 | EXIT_EMA400 | -14.05 |
| SELL | 2026-04-09 10:15:00 | 410.35 | 2026-04-21 09:15:00 | 424.80 | EXIT_EMA400 | -14.45 |
| SELL | 2026-04-20 13:15:00 | 418.50 | 2026-04-21 09:15:00 | 424.80 | EXIT_EMA400 | -6.30 |
