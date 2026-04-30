# IRCON International Ltd. (IRCON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 152.51
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -32.62
- **Avg P&L per closed trade:** -8.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 263.75 | 273.92 | 273.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 261.85 | 273.79 | 273.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 220.34 | 218.85 | 232.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 212.36 | 218.79 | 232.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 15:15:00 | 218.89 | 206.31 | 217.37 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 189.66 | 166.60 | 166.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 190.92 | 171.82 | 169.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 194.39 | 194.58 | 184.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 15:15:00 | 199.20 | 194.44 | 185.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 191.15 | 196.76 | 190.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 190.25 | 196.53 | 190.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 175.00 | 187.99 | 188.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 173.62 | 186.57 | 187.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-03 14:15:00 | 172.03 | 173.12 | 178.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 185.63 | 172.48 | 176.58 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 178.04 | 166.29 | 166.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 161.10 | 166.45 | 166.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 158.47 | 166.21 | 166.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 164.17 | 162.86 | 164.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 10:15:00 | 160.97 | 162.86 | 164.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 163.97 | 162.79 | 164.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 10:15:00 | 164.51 | 162.81 | 164.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-04 09:15:00 | 212.36 | 2024-12-02 15:15:00 | 218.89 | EXIT_EMA400 | -6.53 |
| BUY | 2025-06-20 15:15:00 | 199.20 | 2025-07-14 12:15:00 | 190.25 | EXIT_EMA400 | -8.95 |
| SELL | 2025-09-03 14:15:00 | 172.03 | 2025-09-15 09:15:00 | 185.63 | EXIT_EMA400 | -13.60 |
| SELL | 2026-01-29 10:15:00 | 160.97 | 2026-01-30 10:15:00 | 164.51 | EXIT_EMA400 | -3.54 |
