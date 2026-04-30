# Sammaan Capital Ltd. (SAMMAANCAP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-07-26 09:15:00 → 2026-04-30 15:15:00 (3029 bars)
- **Last close:** 144.61
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 11.64
- **Avg P&L per closed trade:** 2.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 11:15:00 | 166.29 | 154.19 | 154.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 167.55 | 156.40 | 155.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 157.56 | 158.40 | 156.66 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 152.28 | 155.48 | 155.49 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 160.62 | 155.54 | 155.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 162.21 | 155.61 | 155.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 156.12 | 156.74 | 156.18 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 141.42 | 155.67 | 155.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 139.87 | 152.58 | 153.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 11:15:00 | 150.97 | 149.39 | 151.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 13:15:00 | 147.57 | 149.39 | 151.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 121.63 | 113.51 | 121.49 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 127.25 | 122.69 | 122.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 128.06 | 122.83 | 122.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 123.83 | 124.64 | 123.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 124.80 | 123.94 | 123.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 09:15:00 | 126.80 | 130.03 | 127.08 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 116.72 | 126.95 | 126.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 115.56 | 126.83 | 126.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 125.38 | 124.52 | 125.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 13:15:00 | 120.90 | 124.22 | 125.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-29 12:15:00 | 125.75 | 123.44 | 124.83 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 138.44 | 126.00 | 125.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 139.33 | 127.19 | 126.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 171.00 | 174.36 | 162.71 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 149.13 | 157.23 | 157.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 147.10 | 156.10 | 156.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 149.30 | 149.11 | 152.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 10:15:00 | 148.43 | 149.23 | 152.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 147.16 | 144.31 | 147.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-29 15:15:00 | 148.40 | 144.38 | 147.95 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 154.00 | 148.88 | 148.87 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 145.96 | 148.84 | 148.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 141.97 | 148.78 | 148.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 147.25 | 147.04 | 147.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-12 09:15:00 | 142.60 | 146.97 | 147.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 09:15:00 | 153.70 | 143.03 | 145.35 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 154.10 | 146.79 | 146.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 156.64 | 146.95 | 146.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 146.14 | 148.60 | 147.78 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 141.22 | 147.09 | 147.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 141.00 | 147.02 | 147.08 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-07 13:15:00 | 147.57 | 2025-02-11 15:15:00 | 134.65 | TARGET | 12.92 |
| BUY | 2025-06-24 09:15:00 | 124.80 | 2025-06-25 09:15:00 | 128.64 | TARGET | 3.84 |
| SELL | 2025-08-25 13:15:00 | 120.90 | 2025-08-29 12:15:00 | 125.75 | EXIT_EMA400 | -4.85 |
| SELL | 2026-01-07 10:15:00 | 148.43 | 2026-01-20 15:15:00 | 137.61 | TARGET | 10.82 |
| SELL | 2026-03-12 09:15:00 | 142.60 | 2026-03-25 09:15:00 | 153.70 | EXIT_EMA400 | -11.10 |
