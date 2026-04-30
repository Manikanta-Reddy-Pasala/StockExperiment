# Piramal Pharma Ltd. (PPLPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 162.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 4.82
- **Avg P&L per closed trade:** 1.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 237.35 | 245.70 | 245.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 236.20 | 245.36 | 245.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 11:15:00 | 223.53 | 220.81 | 230.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-21 13:15:00 | 208.83 | 220.44 | 229.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 217.14 | 208.76 | 218.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 14:15:00 | 219.03 | 208.94 | 218.17 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 201.11 | 198.52 | 198.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 203.42 | 198.82 | 198.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 198.79 | 198.88 | 198.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-06 09:15:00 | 206.19 | 199.01 | 198.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 206.19 | 199.01 | 198.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 14:15:00 | 197.79 | 199.16 | 198.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 194.30 | 198.57 | 198.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 193.66 | 198.12 | 198.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 178.21 | 177.60 | 183.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 175.35 | 178.18 | 183.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 151.45 | 146.86 | 152.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 10:15:00 | 152.72 | 146.92 | 152.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-21 13:15:00 | 208.83 | 2025-03-19 14:15:00 | 219.03 | EXIT_EMA400 | -10.20 |
| BUY | 2025-11-06 09:15:00 | 206.19 | 2025-11-06 14:15:00 | 197.79 | EXIT_EMA400 | -8.40 |
| SELL | 2026-01-08 10:15:00 | 175.35 | 2026-01-23 12:15:00 | 151.93 | TARGET | 23.42 |
