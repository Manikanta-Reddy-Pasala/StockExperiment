# Indian Energy Exchange Ltd. (IEX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 125.33
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -8.33
- **Avg P&L per closed trade:** -4.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 10:15:00 | 183.23 | 197.06 | 197.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 179.35 | 196.17 | 196.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 11:15:00 | 175.40 | 175.29 | 182.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 14:15:00 | 173.25 | 180.27 | 182.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 10:15:00 | 177.51 | 172.68 | 176.50 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 12:15:00 | 172.00 | 170.73 | 170.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 14:15:00 | 173.80 | 170.77 | 170.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 193.49 | 197.77 | 191.16 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 143.63 | 191.33 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 140.38 | 190.37 | 191.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 147.28 | 146.60 | 156.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 144.08 | 147.12 | 155.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-24 11:15:00 | 148.15 | 140.97 | 146.98 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 14:15:00 | 173.25 | 2025-01-29 10:15:00 | 177.51 | EXIT_EMA400 | -4.26 |
| SELL | 2025-09-23 10:15:00 | 144.08 | 2025-10-24 11:15:00 | 148.15 | EXIT_EMA400 | -4.07 |
