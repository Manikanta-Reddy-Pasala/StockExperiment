# Indian Oil Corporation Ltd. (IOC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 142.44
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 29.36
- **Avg P&L per closed trade:** 4.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 164.99 | 170.97 | 170.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 164.45 | 170.85 | 170.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 142.75 | 142.35 | 149.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 14:15:00 | 141.45 | 142.47 | 148.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 125.30 | 122.40 | 126.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-12 11:15:00 | 124.70 | 122.45 | 126.92 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 126.37 | 123.03 | 126.69 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 126.95 | 123.07 | 126.69 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 133.38 | 128.53 | 128.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 14:15:00 | 133.74 | 128.58 | 128.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 141.12 | 141.13 | 137.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 142.21 | 140.96 | 138.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 140.80 | 141.58 | 138.74 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 141.10 | 141.57 | 138.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 138.80 | 141.40 | 139.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 139.76 | 143.82 | 143.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 139.40 | 143.75 | 143.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 142.94 | 141.63 | 142.47 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 148.69 | 143.12 | 143.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 148.91 | 144.38 | 143.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 149.94 | 150.17 | 147.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-27 09:15:00 | 153.71 | 150.23 | 147.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 160.98 | 163.99 | 160.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-19 14:15:00 | 162.61 | 163.87 | 160.77 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 161.08 | 163.70 | 161.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-26 09:15:00 | 160.06 | 163.66 | 161.00 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 146.50 | 166.55 | 166.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 144.47 | 165.09 | 165.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 147.78 | 147.48 | 153.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 13:15:00 | 145.56 | 147.40 | 153.05 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-12 14:15:00 | 141.45 | 2025-02-12 09:15:00 | 119.25 | TARGET | 22.20 |
| SELL | 2025-03-12 11:15:00 | 124.70 | 2025-03-19 10:15:00 | 126.95 | EXIT_EMA400 | -2.25 |
| BUY | 2025-06-09 09:15:00 | 142.21 | 2025-06-19 12:15:00 | 138.80 | EXIT_EMA400 | -3.41 |
| BUY | 2025-06-13 10:15:00 | 141.10 | 2025-06-19 12:15:00 | 138.80 | EXIT_EMA400 | -2.30 |
| BUY | 2025-10-27 09:15:00 | 153.71 | 2025-11-10 09:15:00 | 171.37 | TARGET | 17.66 |
| BUY | 2025-12-19 14:15:00 | 162.61 | 2025-12-26 09:15:00 | 160.06 | EXIT_EMA400 | -2.55 |
