# Federal Bank Ltd. (FEDERALBNK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 286.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 2.62
- **Avg P&L per closed trade:** 0.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 13:15:00 | 145.35 | 149.33 | 149.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 145.15 | 149.22 | 149.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 10:15:00 | 149.10 | 148.42 | 148.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-08 14:15:00 | 146.85 | 148.47 | 148.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 148.15 | 148.09 | 148.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-14 14:15:00 | 149.15 | 148.10 | 148.58 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 165.10 | 149.12 | 149.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 169.30 | 154.46 | 152.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 14:15:00 | 156.80 | 156.90 | 154.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-10 09:15:00 | 159.45 | 156.92 | 154.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 153.35 | 160.51 | 157.93 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 187.54 | 200.97 | 201.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 178.95 | 195.75 | 197.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 183.24 | 181.44 | 185.86 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 190.75 | 188.37 | 188.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 191.89 | 188.46 | 188.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 190.88 | 193.95 | 191.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 196.10 | 192.45 | 191.29 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 206.79 | 210.78 | 207.27 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 197.62 | 204.89 | 204.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 12:15:00 | 197.09 | 204.74 | 204.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 12:15:00 | 196.64 | 196.44 | 199.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-11 09:15:00 | 195.36 | 196.42 | 199.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 198.87 | 196.42 | 198.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 15:15:00 | 198.98 | 196.45 | 198.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 213.99 | 199.04 | 199.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 215.64 | 199.21 | 199.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 258.20 | 259.39 | 247.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 12:15:00 | 271.50 | 257.05 | 249.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 271.70 | 286.92 | 276.64 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-08 14:15:00 | 146.85 | 2024-02-14 14:15:00 | 149.15 | EXIT_EMA400 | -2.30 |
| BUY | 2024-05-10 09:15:00 | 159.45 | 2024-06-04 10:15:00 | 153.35 | EXIT_EMA400 | -6.10 |
| BUY | 2025-05-13 09:15:00 | 196.10 | 2025-06-03 11:15:00 | 210.54 | TARGET | 14.44 |
| SELL | 2025-09-11 09:15:00 | 195.36 | 2025-09-17 15:15:00 | 198.98 | EXIT_EMA400 | -3.62 |
| BUY | 2026-01-16 12:15:00 | 271.50 | 2026-03-09 09:15:00 | 271.70 | EXIT_EMA400 | 0.20 |
