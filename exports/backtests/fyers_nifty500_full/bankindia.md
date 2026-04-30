# Bank of India (BANKINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 139.94
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 0.77
- **Avg P&L per closed trade:** 0.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 115.93 | 110.26 | 110.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 102.98 | 110.44 | 110.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 12:15:00 | 102.60 | 110.14 | 110.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 102.76 | 102.15 | 105.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 15:15:00 | 101.49 | 102.14 | 105.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 104.10 | 101.56 | 104.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-27 14:15:00 | 104.95 | 101.59 | 104.31 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 111.67 | 102.54 | 102.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 112.50 | 102.99 | 102.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 111.95 | 112.04 | 108.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 114.25 | 111.47 | 108.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 116.10 | 119.95 | 116.24 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 112.08 | 115.70 | 115.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 111.60 | 115.55 | 115.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 113.88 | 113.75 | 114.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 12:15:00 | 112.96 | 113.71 | 114.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.48 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 117.40 | 114.59 | 114.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 118.32 | 114.76 | 114.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 116.43 | 116.98 | 115.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 118.98 | 117.00 | 115.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 139.88 | 141.41 | 138.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-30 11:15:00 | 140.53 | 141.38 | 138.15 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 151.81 | 153.94 | 147.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-03 09:15:00 | 156.65 | 153.75 | 147.98 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 159.39 | 165.65 | 159.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 150.01 | 165.49 | 159.15 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 144.97 | 155.47 | 155.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 155.33 | 155.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 150.11 | 149.42 | 151.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-17 10:15:00 | 147.36 | 149.38 | 151.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 10:15:00 | 152.72 | 149.42 | 151.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-20 15:15:00 | 101.49 | 2025-01-27 14:15:00 | 104.95 | EXIT_EMA400 | -3.46 |
| BUY | 2025-05-13 09:15:00 | 114.25 | 2025-06-19 10:15:00 | 116.10 | EXIT_EMA400 | 1.85 |
| SELL | 2025-08-14 12:15:00 | 112.96 | 2025-08-18 09:15:00 | 114.75 | EXIT_EMA400 | -1.79 |
| BUY | 2025-09-29 09:15:00 | 118.98 | 2025-10-07 09:15:00 | 128.02 | TARGET | 9.04 |
| BUY | 2025-12-30 11:15:00 | 140.53 | 2026-01-01 15:15:00 | 147.66 | TARGET | 7.13 |
| BUY | 2026-02-03 09:15:00 | 156.65 | 2026-03-09 09:15:00 | 150.01 | EXIT_EMA400 | -6.64 |
| SELL | 2026-04-17 10:15:00 | 147.36 | 2026-04-22 10:15:00 | 152.72 | EXIT_EMA400 | -5.36 |
