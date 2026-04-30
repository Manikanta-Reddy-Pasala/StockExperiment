# JM Financial Ltd. (JMFINANCIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 138.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 23.70
- **Avg P&L per closed trade:** 5.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 123.00 | 133.33 | 133.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 121.23 | 133.10 | 133.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 117.09 | 115.64 | 121.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 111.81 | 115.65 | 121.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 100.33 | 95.79 | 101.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-16 12:15:00 | 99.71 | 95.83 | 101.04 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 99.95 | 95.98 | 101.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 09:15:00 | 102.39 | 96.29 | 100.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 115.95 | 102.79 | 102.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 118.04 | 103.31 | 103.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 161.15 | 162.16 | 148.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 176.94 | 159.60 | 151.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 173.21 | 178.53 | 170.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-25 10:15:00 | 169.41 | 177.20 | 170.63 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 150.01 | 169.54 | 169.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 148.66 | 169.14 | 169.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 156.93 | 154.25 | 160.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 146.11 | 153.82 | 159.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 137.82 | 132.61 | 139.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 10:15:00 | 140.60 | 132.69 | 139.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 138.78 | 131.28 | 131.26 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 111.81 | 2025-04-07 09:15:00 | 83.41 | TARGET | 28.40 |
| SELL | 2025-04-16 12:15:00 | 99.71 | 2025-04-21 09:15:00 | 102.39 | EXIT_EMA400 | -2.68 |
| BUY | 2025-08-13 09:15:00 | 176.94 | 2025-09-25 10:15:00 | 169.41 | EXIT_EMA400 | -7.53 |
| SELL | 2025-12-04 09:15:00 | 146.11 | 2026-02-10 10:15:00 | 140.60 | EXIT_EMA400 | 5.51 |
