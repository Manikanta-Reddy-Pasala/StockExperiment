# Piramal Pharma Ltd. (PPLPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 161.87
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -11.10
- **Avg P&L per closed trade:** -1.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 88.20 | 98.59 | 98.62 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 106.20 | 98.59 | 98.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 11:15:00 | 107.75 | 98.77 | 98.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 135.15 | 135.79 | 127.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 13:15:00 | 139.25 | 135.89 | 127.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 134.25 | 138.60 | 132.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-12 13:15:00 | 134.95 | 138.47 | 132.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-02-12 14:15:00 | 132.95 | 138.42 | 132.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 14:15:00 | 118.40 | 131.79 | 131.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 117.00 | 130.54 | 131.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 127.60 | 126.41 | 128.61 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 141.60 | 130.23 | 130.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 11:15:00 | 142.30 | 130.69 | 130.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 145.70 | 146.74 | 142.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-31 13:15:00 | 147.80 | 146.76 | 142.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 140.05 | 146.91 | 142.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 236.20 | 245.35 | 245.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 234.60 | 245.02 | 245.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 11:15:00 | 223.54 | 221.00 | 230.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-21 13:15:00 | 208.83 | 220.61 | 229.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 217.20 | 208.83 | 218.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 14:15:00 | 218.96 | 209.01 | 218.26 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 202.09 | 198.51 | 198.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 202.30 | 198.68 | 198.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 198.79 | 198.88 | 198.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-06 09:15:00 | 206.14 | 199.01 | 198.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 206.14 | 199.01 | 198.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 14:15:00 | 197.79 | 199.16 | 198.85 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 194.30 | 198.57 | 198.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 193.66 | 198.12 | 198.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 178.21 | 177.61 | 183.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 175.35 | 178.18 | 183.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 10:15:00 | 152.72 | 146.85 | 152.27 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-18 13:15:00 | 139.25 | 2024-02-12 14:15:00 | 132.95 | EXIT_EMA400 | -6.30 |
| BUY | 2024-02-12 13:15:00 | 134.95 | 2024-02-12 14:15:00 | 132.95 | EXIT_EMA400 | -2.00 |
| BUY | 2024-05-31 13:15:00 | 147.80 | 2024-06-04 11:15:00 | 140.05 | EXIT_EMA400 | -7.75 |
| SELL | 2025-02-21 13:15:00 | 208.83 | 2025-03-19 14:15:00 | 218.96 | EXIT_EMA400 | -10.13 |
| BUY | 2025-11-06 09:15:00 | 206.14 | 2025-11-06 14:15:00 | 197.79 | EXIT_EMA400 | -8.35 |
| SELL | 2026-01-08 10:15:00 | 175.35 | 2026-01-23 12:15:00 | 151.92 | TARGET | 23.43 |
