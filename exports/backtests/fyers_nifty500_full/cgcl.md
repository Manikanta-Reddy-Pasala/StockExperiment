# Capri Global Capital Ltd. (CGCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 186.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 39.33
- **Avg P&L per closed trade:** 6.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 205.90 | 203.07 | 203.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 206.35 | 203.10 | 203.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 13:15:00 | 202.85 | 203.26 | 203.16 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 200.34 | 203.05 | 203.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 197.99 | 202.95 | 203.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 09:15:00 | 198.66 | 186.27 | 191.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 186.10 | 187.15 | 191.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 186.10 | 187.15 | 191.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-27 10:15:00 | 184.40 | 187.12 | 191.83 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 176.98 | 175.06 | 182.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-21 09:15:00 | 173.31 | 175.07 | 181.87 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 167.30 | 166.24 | 172.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-26 11:15:00 | 185.26 | 166.46 | 172.37 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 174.90 | 166.50 | 166.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 177.95 | 166.61 | 166.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 14:15:00 | 167.90 | 168.17 | 167.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 11:15:00 | 170.30 | 167.73 | 167.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-09 15:15:00 | 167.00 | 167.88 | 167.38 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 180.01 | 190.91 | 190.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 179.29 | 188.36 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 10:15:00 | 180.12 | 184.57 | 186.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 179.60 | 175.78 | 179.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 09:15:00 | 175.80 | 175.83 | 179.80 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-12 12:15:00 | 179.60 | 175.92 | 179.52 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 182.00 | 173.11 | 173.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 185.50 | 173.23 | 173.17 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-27 09:15:00 | 186.10 | 2025-01-29 09:15:00 | 168.78 | TARGET | 17.32 |
| SELL | 2025-01-27 10:15:00 | 184.40 | 2025-02-12 14:15:00 | 162.10 | TARGET | 22.30 |
| SELL | 2025-02-21 09:15:00 | 173.31 | 2025-03-26 11:15:00 | 185.26 | EXIT_EMA400 | -11.95 |
| BUY | 2025-07-08 11:15:00 | 170.30 | 2025-07-09 15:15:00 | 167.00 | EXIT_EMA400 | -3.30 |
| SELL | 2026-01-12 10:15:00 | 180.12 | 2026-01-27 09:15:00 | 161.36 | TARGET | 18.76 |
| SELL | 2026-02-10 09:15:00 | 175.80 | 2026-02-12 12:15:00 | 179.60 | EXIT_EMA400 | -3.80 |
