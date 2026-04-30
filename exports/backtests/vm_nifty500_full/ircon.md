# IRCON International Ltd. (IRCON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 152.20
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
- **Total realized P&L (per unit):** -28.89
- **Avg P&L per closed trade:** -7.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 262.65 | 274.41 | 274.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 261.95 | 273.85 | 274.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 217.78 | 217.78 | 230.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 13:15:00 | 216.39 | 217.77 | 230.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 217.22 | 206.30 | 217.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 219.19 | 206.42 | 217.47 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 189.66 | 166.60 | 166.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 190.92 | 171.82 | 169.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 194.32 | 194.58 | 184.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 15:15:00 | 199.20 | 194.44 | 185.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 191.15 | 196.76 | 190.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 190.25 | 196.53 | 190.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 175.00 | 187.98 | 188.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 173.62 | 186.57 | 187.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 177.67 | 173.05 | 178.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-03 14:15:00 | 172.03 | 173.11 | 178.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 185.63 | 172.47 | 176.57 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 178.04 | 166.30 | 166.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 161.10 | 166.45 | 166.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 160.77 | 166.29 | 166.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 164.18 | 162.86 | 164.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 10:15:00 | 160.97 | 162.85 | 164.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 10:15:00 | 164.51 | 162.80 | 164.36 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 13:15:00 | 216.39 | 2024-12-03 09:15:00 | 219.19 | EXIT_EMA400 | -2.80 |
| BUY | 2025-06-20 15:15:00 | 199.20 | 2025-07-14 12:15:00 | 190.25 | EXIT_EMA400 | -8.95 |
| SELL | 2025-09-03 14:15:00 | 172.03 | 2025-09-15 09:15:00 | 185.63 | EXIT_EMA400 | -13.60 |
| SELL | 2026-01-29 10:15:00 | 160.97 | 2026-01-30 10:15:00 | 164.51 | EXIT_EMA400 | -3.54 |
