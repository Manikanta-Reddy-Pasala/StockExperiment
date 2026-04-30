# Bank of India (BANKINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 139.87
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -11.91
- **Avg P&L per closed trade:** -1.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 123.15 | 136.83 | 136.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 118.65 | 133.53 | 134.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 123.50 | 123.41 | 127.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-10 09:15:00 | 120.49 | 123.43 | 127.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-29 11:15:00 | 125.95 | 122.15 | 125.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 12:15:00 | 115.63 | 110.52 | 110.49 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 103.79 | 110.58 | 110.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 102.87 | 110.50 | 110.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 102.76 | 102.15 | 105.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 15:15:00 | 101.46 | 102.14 | 105.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 104.03 | 101.56 | 104.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-27 14:15:00 | 104.95 | 101.59 | 104.34 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 111.67 | 102.52 | 102.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 112.50 | 102.97 | 102.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 111.95 | 112.03 | 108.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 114.25 | 111.47 | 108.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 116.10 | 119.95 | 116.23 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 112.08 | 115.70 | 115.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 111.67 | 115.54 | 115.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 113.87 | 113.75 | 114.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 12:15:00 | 112.96 | 113.71 | 114.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.47 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 117.40 | 114.59 | 114.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 118.32 | 114.76 | 114.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 116.45 | 116.98 | 115.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 118.98 | 117.00 | 115.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 151.81 | 153.90 | 147.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-03 09:15:00 | 156.65 | 153.72 | 147.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 149.95 | 165.50 | 159.05 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 141.72 | 155.32 | 155.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 139.90 | 155.17 | 155.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 150.11 | 149.40 | 151.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-17 10:15:00 | 147.37 | 149.37 | 151.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 10:15:00 | 152.72 | 149.39 | 151.51 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-07-10 09:15:00 | 120.49 | 2024-07-29 11:15:00 | 125.95 | EXIT_EMA400 | -5.46 |
| SELL | 2025-01-20 15:15:00 | 101.46 | 2025-01-27 14:15:00 | 104.95 | EXIT_EMA400 | -3.49 |
| BUY | 2025-05-13 09:15:00 | 114.25 | 2025-06-19 10:15:00 | 116.10 | EXIT_EMA400 | 1.85 |
| SELL | 2025-08-14 12:15:00 | 112.96 | 2025-08-18 09:15:00 | 114.75 | EXIT_EMA400 | -1.79 |
| BUY | 2025-09-29 09:15:00 | 118.98 | 2025-10-07 09:15:00 | 128.01 | TARGET | 9.03 |
| BUY | 2026-02-03 09:15:00 | 156.65 | 2026-03-09 09:15:00 | 149.95 | EXIT_EMA400 | -6.70 |
| SELL | 2026-04-17 10:15:00 | 147.37 | 2026-04-22 10:15:00 | 152.72 | EXIT_EMA400 | -5.35 |
