# Canara Bank (CANBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 134.65
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
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 4 / 0
- **Target hits / EMA400 exits:** 3 / 1
- **Total realized P&L (per unit):** 15.89
- **Avg P&L per closed trade:** 3.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 112.90 | 116.57 | 116.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 10:15:00 | 112.44 | 116.49 | 116.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 116.75 | 115.76 | 116.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-31 13:15:00 | 114.42 | 115.76 | 116.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-02 13:15:00 | 112.98 | 111.37 | 112.81 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 99.50 | 91.22 | 91.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 99.98 | 91.47 | 91.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 92.35 | 93.97 | 92.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 94.24 | 93.92 | 92.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 104.96 | 109.56 | 104.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-20 09:15:00 | 105.92 | 109.39 | 104.69 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 109.81 | 112.66 | 109.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-22 11:15:00 | 109.20 | 112.63 | 109.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 107.72 | 108.72 | 108.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 107.48 | 108.69 | 108.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 109.71 | 108.42 | 108.56 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 112.47 | 108.70 | 108.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 113.15 | 109.21 | 108.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 143.27 | 143.62 | 135.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 144.52 | 143.60 | 135.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 148.10 | 152.17 | 147.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-30 13:15:00 | 147.35 | 152.01 | 147.56 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 134.70 | 147.53 | 147.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 134.23 | 147.40 | 147.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 137.19 | 137.17 | 141.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 134.12 | 139.25 | 140.96 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-07-31 13:15:00 | 114.42 | 2024-08-05 09:15:00 | 109.33 | TARGET | 5.09 |
| BUY | 2025-05-08 09:15:00 | 94.24 | 2025-05-12 09:15:00 | 98.52 | TARGET | 4.28 |
| BUY | 2025-06-20 09:15:00 | 105.92 | 2025-06-24 09:15:00 | 109.61 | TARGET | 3.69 |
| BUY | 2025-12-09 10:15:00 | 144.52 | 2026-01-30 13:15:00 | 147.35 | EXIT_EMA400 | 2.83 |
