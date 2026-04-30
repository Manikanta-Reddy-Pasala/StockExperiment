# NBCC (India) Ltd. (NBCC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 91.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -1.16
- **Avg P&L per closed trade:** -0.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 108.86 | 115.76 | 115.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 104.95 | 115.22 | 115.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 100.00 | 97.60 | 103.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 09:15:00 | 96.80 | 97.67 | 102.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 102.82 | 98.07 | 102.37 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 101.48 | 86.46 | 86.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 102.69 | 92.93 | 90.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 116.00 | 116.13 | 108.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 12:15:00 | 117.17 | 116.08 | 108.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 112.68 | 117.74 | 112.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-11 11:15:00 | 113.17 | 117.69 | 112.67 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 112.80 | 117.51 | 112.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 09:15:00 | 113.64 | 117.47 | 112.68 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-14 13:15:00 | 112.65 | 117.30 | 112.69 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 105.76 | 111.40 | 111.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 103.92 | 111.27 | 111.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 105.80 | 105.25 | 107.50 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 112.15 | 108.57 | 108.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 112.45 | 108.61 | 108.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 109.50 | 109.77 | 109.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 10:15:00 | 110.70 | 109.78 | 109.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 110.37 | 110.73 | 109.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-29 09:15:00 | 112.09 | 110.74 | 109.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-07 09:15:00 | 110.63 | 112.41 | 111.02 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 103.75 | 113.34 | 113.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 102.61 | 113.13 | 113.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 88.10 | 86.54 | 92.43 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-29 09:15:00 | 96.80 | 2024-12-05 12:15:00 | 102.82 | EXIT_EMA400 | -6.02 |
| BUY | 2025-06-20 12:15:00 | 117.17 | 2025-07-14 13:15:00 | 112.65 | EXIT_EMA400 | -4.52 |
| BUY | 2025-07-11 11:15:00 | 113.17 | 2025-07-14 13:15:00 | 112.65 | EXIT_EMA400 | -0.52 |
| BUY | 2025-07-14 09:15:00 | 113.64 | 2025-07-14 13:15:00 | 112.65 | EXIT_EMA400 | -0.99 |
| BUY | 2025-10-15 10:15:00 | 110.70 | 2025-10-29 12:15:00 | 115.10 | TARGET | 4.40 |
| BUY | 2025-10-29 09:15:00 | 112.09 | 2025-10-30 10:15:00 | 118.59 | TARGET | 6.50 |
