# Manappuram Finance Ltd. (MANAPPURAM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 296.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -8.29
- **Avg P&L per closed trade:** -2.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 186.27 | 204.04 | 204.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 184.16 | 203.84 | 204.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 163.08 | 159.45 | 169.95 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 189.49 | 174.18 | 174.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 190.10 | 176.28 | 175.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 177.70 | 178.93 | 176.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-08 14:15:00 | 180.03 | 178.94 | 176.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 176.95 | 179.19 | 177.15 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 267.50 | 294.08 | 294.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 264.80 | 293.52 | 293.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 269.50 | 266.27 | 275.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 262.60 | 266.31 | 275.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 271.20 | 266.31 | 274.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-15 12:15:00 | 268.20 | 266.41 | 274.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 274.20 | 267.09 | 273.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-08 14:15:00 | 180.03 | 2025-01-09 09:15:00 | 189.34 | TARGET | 9.31 |
| SELL | 2026-04-09 13:15:00 | 262.60 | 2026-04-21 09:15:00 | 274.20 | EXIT_EMA400 | -11.60 |
| SELL | 2026-04-15 12:15:00 | 268.20 | 2026-04-21 09:15:00 | 274.20 | EXIT_EMA400 | -6.00 |
