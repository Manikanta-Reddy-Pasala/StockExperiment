# Bandhan Bank Ltd. (BANDHANBNK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 201.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 4 |
| EXIT | 3 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 55.05
- **Avg P&L per closed trade:** 7.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 185.48 | 199.69 | 199.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 185.11 | 196.57 | 197.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 177.08 | 175.97 | 182.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 09:15:00 | 174.67 | 176.14 | 182.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-07 10:15:00 | 150.82 | 144.28 | 150.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 168.90 | 150.05 | 150.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 170.61 | 150.64 | 150.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 157.31 | 157.70 | 154.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 158.60 | 157.69 | 154.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 157.00 | 157.87 | 154.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 11:15:00 | 157.27 | 157.84 | 154.93 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 175.17 | 179.67 | 175.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 166.72 | 172.71 | 172.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 166.19 | 172.59 | 172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 168.23 | 172.04 | 172.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 167.79 | 166.70 | 168.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-18 11:15:00 | 166.91 | 166.71 | 168.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 165.18 | 164.14 | 166.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-06 12:15:00 | 164.61 | 164.17 | 166.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 165.36 | 164.21 | 166.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 14:15:00 | 164.39 | 164.24 | 166.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-10 12:15:00 | 166.58 | 163.98 | 166.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 167.15 | 151.07 | 151.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 168.54 | 151.85 | 151.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 169.12 | 171.80 | 164.89 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 142.15 | 160.77 | 160.79 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 170.67 | 160.63 | 160.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 171.55 | 160.74 | 160.66 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-06 09:15:00 | 174.67 | 2025-01-06 14:15:00 | 152.05 | TARGET | 22.62 |
| BUY | 2025-05-09 11:15:00 | 157.27 | 2025-05-12 09:15:00 | 164.30 | TARGET | 7.03 |
| BUY | 2025-05-07 10:15:00 | 158.60 | 2025-05-16 09:15:00 | 170.53 | TARGET | 11.93 |
| SELL | 2025-09-18 11:15:00 | 166.91 | 2025-09-23 09:15:00 | 161.42 | TARGET | 5.49 |
| SELL | 2025-08-26 09:15:00 | 168.23 | 2025-09-26 09:15:00 | 156.09 | TARGET | 12.14 |
| SELL | 2025-10-06 12:15:00 | 164.61 | 2025-10-10 12:15:00 | 166.58 | EXIT_EMA400 | -1.97 |
| SELL | 2025-10-07 14:15:00 | 164.39 | 2025-10-10 12:15:00 | 166.58 | EXIT_EMA400 | -2.19 |
