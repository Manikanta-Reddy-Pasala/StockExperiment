# Indraprastha Gas Ltd. (IGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 165.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 1.90
- **Avg P&L per closed trade:** 0.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 223.55 | 264.29 | 264.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 222.00 | 263.87 | 264.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 192.80 | 191.89 | 213.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 189.15 | 191.92 | 212.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 205.53 | 194.66 | 205.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-31 11:15:00 | 206.15 | 194.77 | 205.71 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 204.20 | 193.51 | 193.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 206.60 | 194.22 | 193.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 205.17 | 205.97 | 201.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 207.25 | 205.32 | 201.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 202.06 | 205.74 | 202.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 204.96 | 208.06 | 208.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 204.53 | 208.03 | 208.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 217.03 | 207.92 | 207.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 217.52 | 208.10 | 208.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 209.80 | 210.15 | 209.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 12:15:00 | 211.58 | 210.17 | 209.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-09 14:15:00 | 208.93 | 210.15 | 209.17 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 203.78 | 210.90 | 210.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 202.21 | 210.81 | 210.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 196.90 | 196.22 | 201.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 09:15:00 | 192.54 | 196.19 | 201.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 164.45 | 157.03 | 164.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 164.71 | 157.10 | 164.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 09:15:00 | 189.15 | 2024-12-31 11:15:00 | 206.15 | EXIT_EMA400 | -17.00 |
| BUY | 2025-06-16 10:15:00 | 207.25 | 2025-06-19 12:15:00 | 202.06 | EXIT_EMA400 | -5.19 |
| BUY | 2025-09-09 12:15:00 | 211.58 | 2025-09-09 14:15:00 | 208.93 | EXIT_EMA400 | -2.65 |
| SELL | 2025-12-19 09:15:00 | 192.54 | 2026-02-13 09:15:00 | 165.80 | TARGET | 26.74 |
