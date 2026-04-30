# Castrol India Ltd. (CASTROLIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 185.17
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 26.82
- **Avg P&L per closed trade:** 4.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 13:15:00 | 229.54 | 246.82 | 246.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 226.84 | 245.94 | 246.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 208.60 | 205.81 | 217.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 203.73 | 210.13 | 215.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 199.70 | 185.48 | 194.59 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 217.14 | 199.81 | 199.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 219.31 | 201.00 | 200.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 218.50 | 218.52 | 211.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-20 09:15:00 | 219.64 | 218.53 | 211.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 212.25 | 218.40 | 212.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-03-25 11:15:00 | 210.89 | 218.32 | 212.06 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 11:15:00 | 201.85 | 208.38 | 208.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 201.27 | 207.07 | 207.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 15:15:00 | 204.70 | 204.43 | 205.98 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 218.70 | 207.03 | 206.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 208.66 | 207.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 09:15:00 | 213.91 | 211.37 | 210.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 218.66 | 220.34 | 217.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-01 09:15:00 | 222.39 | 220.27 | 217.26 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 217.75 | 220.24 | 217.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 09:15:00 | 219.76 | 220.23 | 217.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 217.95 | 220.27 | 217.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-06 11:15:00 | 215.50 | 220.20 | 217.55 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 206.55 | 215.56 | 215.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.51 | 213.36 | 214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 10:15:00 | 203.70 | 204.20 | 208.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-15 15:15:00 | 204.55 | 201.57 | 204.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-18 09:15:00 | 203.73 | 2025-01-27 10:15:00 | 167.10 | TARGET | 36.63 |
| BUY | 2025-03-20 09:15:00 | 219.64 | 2025-03-25 11:15:00 | 210.89 | EXIT_EMA400 | -8.75 |
| BUY | 2025-06-26 09:15:00 | 213.91 | 2025-07-01 09:15:00 | 224.85 | TARGET | 10.94 |
| BUY | 2025-08-01 09:15:00 | 222.39 | 2025-08-06 11:15:00 | 215.50 | EXIT_EMA400 | -6.89 |
| BUY | 2025-08-04 09:15:00 | 219.76 | 2025-08-06 11:15:00 | 215.50 | EXIT_EMA400 | -4.26 |
| SELL | 2025-09-18 10:15:00 | 203.70 | 2025-10-15 15:15:00 | 204.55 | EXIT_EMA400 | -0.85 |
