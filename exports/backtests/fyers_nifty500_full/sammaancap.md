# Sammaan Capital Ltd. (SAMMAANCAP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 145.38
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 19 |
| ALERT1 | 14 |
| ALERT2 | 12 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 16.80
- **Avg P&L per closed trade:** 2.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 153.00 | 164.39 | 164.42 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 174.26 | 164.40 | 164.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 15:15:00 | 175.25 | 164.60 | 164.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 164.25 | 166.01 | 165.30 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 159.82 | 164.67 | 164.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 158.24 | 164.34 | 164.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 168.90 | 163.42 | 164.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 09:15:00 | 159.58 | 163.67 | 164.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-20 15:15:00 | 164.00 | 163.22 | 163.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 171.60 | 164.36 | 164.35 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 160.32 | 164.37 | 164.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 159.70 | 164.32 | 164.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 147.80 | 147.78 | 153.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 11:15:00 | 145.95 | 147.79 | 153.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-19 09:15:00 | 150.90 | 144.88 | 150.39 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 166.52 | 154.06 | 154.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 167.60 | 156.39 | 155.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 157.56 | 158.39 | 156.57 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 13:15:00 | 152.86 | 155.37 | 155.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 15:15:00 | 152.70 | 155.32 | 155.35 | Break + close below crossover candle low |

### Cycle 8 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 161.07 | 155.38 | 155.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 162.21 | 155.60 | 155.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 156.14 | 156.73 | 156.12 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 15:15:00 | 140.69 | 155.51 | 155.55 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 11:15:00 | 157.30 | 155.35 | 155.34 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 149.59 | 155.29 | 155.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 145.19 | 155.19 | 155.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 149.00 | 148.94 | 151.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 145.05 | 148.92 | 151.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 121.60 | 113.50 | 121.42 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 127.56 | 122.64 | 122.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 128.06 | 122.83 | 122.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 124.80 | 123.93 | 123.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 09:15:00 | 126.85 | 130.02 | 127.07 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 116.72 | 126.94 | 126.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 115.56 | 126.83 | 126.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 13:15:00 | 120.90 | 124.22 | 125.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-29 12:15:00 | 125.71 | 123.44 | 124.83 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 138.44 | 126.01 | 125.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 139.38 | 127.20 | 126.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 171.00 | 174.36 | 162.71 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 149.13 | 157.24 | 157.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 147.09 | 156.11 | 156.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 149.30 | 149.12 | 152.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-02 15:15:00 | 148.51 | 149.11 | 152.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 151.81 | 149.18 | 152.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 10:15:00 | 149.55 | 149.21 | 152.13 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 147.17 | 144.31 | 147.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-29 15:15:00 | 148.40 | 144.38 | 147.96 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 154.32 | 148.80 | 148.78 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 141.00 | 148.72 | 148.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 138.82 | 147.75 | 148.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-12 09:15:00 | 142.60 | 146.99 | 147.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 09:15:00 | 153.70 | 143.03 | 145.32 | Close above EMA400 |

### Cycle 18 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 154.10 | 146.78 | 146.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 156.65 | 146.95 | 146.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 10:15:00 | 148.76 | 148.77 | 147.84 | EMA200 retest candle locked |

### Cycle 19 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 140.90 | 147.13 | 147.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 139.20 | 147.05 | 147.11 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 09:15:00 | 159.58 | 2024-09-20 15:15:00 | 164.00 | EXIT_EMA400 | -4.42 |
| SELL | 2024-11-07 11:15:00 | 145.95 | 2024-11-19 09:15:00 | 150.90 | EXIT_EMA400 | -4.95 |
| SELL | 2025-02-10 09:15:00 | 145.05 | 2025-02-12 09:15:00 | 125.73 | TARGET | 19.32 |
| BUY | 2025-06-24 09:15:00 | 124.80 | 2025-06-25 09:15:00 | 128.71 | TARGET | 3.91 |
| SELL | 2025-08-25 13:15:00 | 120.90 | 2025-08-29 12:15:00 | 125.71 | EXIT_EMA400 | -4.81 |
| SELL | 2026-01-06 10:15:00 | 149.55 | 2026-01-12 09:15:00 | 141.81 | TARGET | 7.74 |
| SELL | 2026-01-02 15:15:00 | 148.51 | 2026-01-20 15:15:00 | 137.40 | TARGET | 11.11 |
| SELL | 2026-03-12 09:15:00 | 142.60 | 2026-03-25 09:15:00 | 153.70 | EXIT_EMA400 | -11.10 |
