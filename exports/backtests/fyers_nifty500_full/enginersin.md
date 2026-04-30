# Engineers India Ltd. (ENGINERSIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 253.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -10.69
- **Avg P&L per closed trade:** -2.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 223.35 | 250.28 | 250.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 222.55 | 250.00 | 250.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 225.76 | 224.48 | 233.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-13 12:15:00 | 222.88 | 224.55 | 232.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 14:15:00 | 201.13 | 188.94 | 198.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 183.67 | 168.63 | 168.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 15:15:00 | 185.00 | 171.11 | 169.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 171.75 | 172.88 | 171.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 11:15:00 | 175.15 | 172.91 | 171.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 172.00 | 173.21 | 171.31 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 170.30 | 173.18 | 171.30 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 191.77 | 217.05 | 217.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 190.70 | 216.79 | 216.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 210.84 | 204.98 | 209.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 10:15:00 | 201.78 | 206.66 | 208.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-30 09:15:00 | 205.49 | 200.95 | 203.66 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 205.60 | 200.07 | 200.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 206.60 | 200.18 | 200.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 200.41 | 200.52 | 200.30 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 191.01 | 200.01 | 200.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 187.00 | 198.48 | 199.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 182.60 | 181.71 | 188.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 15:15:00 | 180.55 | 181.83 | 187.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-13 09:15:00 | 204.70 | 182.05 | 187.80 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 214.27 | 192.63 | 192.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 219.65 | 193.55 | 193.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 15:15:00 | 201.10 | 201.94 | 197.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 09:15:00 | 203.45 | 201.96 | 197.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 195.82 | 202.26 | 198.35 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 14:15:00 | 187.43 | 196.14 | 196.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 186.21 | 195.43 | 195.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 199.80 | 194.90 | 195.51 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 207.24 | 196.13 | 196.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 210.22 | 196.78 | 196.41 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-13 12:15:00 | 222.88 | 2024-10-07 10:15:00 | 193.23 | TARGET | 29.65 |
| BUY | 2025-05-07 11:15:00 | 175.15 | 2025-05-09 09:15:00 | 170.30 | EXIT_EMA400 | -4.85 |
| SELL | 2025-09-25 10:15:00 | 201.78 | 2025-10-30 09:15:00 | 205.49 | EXIT_EMA400 | -3.71 |
| SELL | 2026-02-12 15:15:00 | 180.55 | 2026-02-13 09:15:00 | 204.70 | EXIT_EMA400 | -24.15 |
| BUY | 2026-03-05 09:15:00 | 203.45 | 2026-03-09 09:15:00 | 195.82 | EXIT_EMA400 | -7.63 |
