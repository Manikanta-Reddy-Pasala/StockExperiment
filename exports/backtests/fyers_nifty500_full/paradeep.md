# Paradeep Phosphates Ltd. (PARADEEP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 128.99
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
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 31.20
- **Avg P&L per closed trade:** 15.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 15:15:00 | 99.15 | 108.56 | 108.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 98.13 | 108.46 | 108.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 95.65 | 95.40 | 99.56 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 13:15:00 | 113.92 | 102.13 | 102.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 15:15:00 | 114.65 | 102.37 | 102.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 163.60 | 163.61 | 151.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 166.70 | 163.64 | 151.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-08 09:15:00 | 194.66 | 209.86 | 196.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 178.60 | 189.02 | 189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 176.42 | 188.70 | 188.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 163.16 | 160.22 | 167.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 153.01 | 159.86 | 166.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-30 09:15:00 | 168.28 | 159.74 | 164.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-24 09:15:00 | 166.70 | 2025-07-29 09:15:00 | 213.17 | TARGET | 46.47 |
| SELL | 2025-12-18 09:15:00 | 153.01 | 2025-12-30 09:15:00 | 168.28 | EXIT_EMA400 | -15.27 |
