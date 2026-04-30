# Capri Global Capital Ltd. (CGCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 186.18
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 9 |
| ENTRY1 | 5 |
| ENTRY2 | 8 |
| EXIT | 5 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / EMA400 exits:** 6 / 7
- **Total realized P&L (per unit):** -141.13
- **Avg P&L per closed trade:** -10.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 379.08 | 389.13 | 389.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 378.83 | 388.77 | 388.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 10:15:00 | 385.70 | 384.36 | 386.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-30 11:15:00 | 379.98 | 384.28 | 386.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 384.40 | 383.76 | 385.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-02 10:15:00 | 381.92 | 383.74 | 385.88 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 382.85 | 383.69 | 385.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-03 14:15:00 | 380.27 | 383.70 | 385.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 380.77 | 383.04 | 385.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-09 14:15:00 | 377.25 | 382.82 | 385.00 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-13 09:15:00 | 385.00 | 382.53 | 384.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 12:15:00 | 386.52 | 382.40 | 382.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 14:15:00 | 388.67 | 382.50 | 382.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 449.48 | 451.04 | 429.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 14:15:00 | 462.85 | 451.47 | 430.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-05 09:15:00 | 280.50 | 469.12 | 449.74 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 248.45 | 431.91 | 432.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 239.40 | 429.99 | 431.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 238.00 | 236.04 | 274.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-27 09:15:00 | 220.75 | 230.45 | 260.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-01 14:15:00 | 225.05 | 214.80 | 223.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 174.90 | 166.51 | 166.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 177.95 | 166.63 | 166.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 14:15:00 | 167.90 | 168.18 | 167.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 11:15:00 | 170.26 | 167.74 | 167.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-09 15:15:00 | 167.00 | 167.88 | 167.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 180.01 | 190.91 | 190.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 179.29 | 188.36 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 185.50 | 183.79 | 186.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 10:15:00 | 180.12 | 184.57 | 186.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 179.60 | 176.16 | 180.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 09:15:00 | 175.80 | 176.19 | 180.13 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 179.60 | 176.22 | 179.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 13:15:00 | 178.27 | 176.24 | 179.82 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 179.30 | 176.30 | 179.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-13 09:15:00 | 175.63 | 176.30 | 179.79 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 176.76 | 176.16 | 179.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-19 10:15:00 | 175.60 | 176.16 | 179.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 170.98 | 168.67 | 172.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-19 09:15:00 | 169.62 | 168.78 | 172.80 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 171.50 | 167.87 | 171.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-01 10:15:00 | 172.80 | 167.50 | 171.20 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 185.47 | 173.28 | 173.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 185.78 | 174.53 | 173.90 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-30 11:15:00 | 379.98 | 2023-11-13 09:15:00 | 385.00 | EXIT_EMA400 | -5.02 |
| SELL | 2023-11-02 10:15:00 | 381.92 | 2023-11-13 09:15:00 | 385.00 | EXIT_EMA400 | -3.08 |
| SELL | 2023-11-03 14:15:00 | 380.27 | 2023-11-13 09:15:00 | 385.00 | EXIT_EMA400 | -4.73 |
| SELL | 2023-11-09 14:15:00 | 377.25 | 2023-11-13 09:15:00 | 385.00 | EXIT_EMA400 | -7.75 |
| BUY | 2024-02-13 14:15:00 | 462.85 | 2024-03-05 09:15:00 | 280.50 | EXIT_EMA400 | -182.35 |
| SELL | 2024-05-27 09:15:00 | 220.75 | 2024-08-01 14:15:00 | 225.05 | EXIT_EMA400 | -4.30 |
| BUY | 2025-07-08 11:15:00 | 170.26 | 2025-07-09 15:15:00 | 167.00 | EXIT_EMA400 | -3.26 |
| SELL | 2026-01-12 10:15:00 | 180.12 | 2026-01-27 09:15:00 | 161.35 | TARGET | 18.77 |
| SELL | 2026-02-12 13:15:00 | 178.27 | 2026-02-17 10:15:00 | 173.63 | TARGET | 4.64 |
| SELL | 2026-02-19 10:15:00 | 175.60 | 2026-02-27 09:15:00 | 164.67 | TARGET | 10.93 |
| SELL | 2026-02-10 09:15:00 | 175.80 | 2026-02-27 14:15:00 | 162.80 | TARGET | 13.00 |
| SELL | 2026-02-13 09:15:00 | 175.63 | 2026-02-27 14:15:00 | 163.15 | TARGET | 12.48 |
| SELL | 2026-03-19 09:15:00 | 169.62 | 2026-03-23 12:15:00 | 160.07 | TARGET | 9.55 |
