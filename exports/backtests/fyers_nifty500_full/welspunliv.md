# Welspun Living Ltd. (WELSPUNLIV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 128.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** -9.00
- **Avg P&L per closed trade:** -1.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 168.41 | 173.71 | 173.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 167.82 | 173.65 | 173.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 15:15:00 | 160.80 | 160.76 | 165.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 158.71 | 160.74 | 165.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 14:15:00 | 159.79 | 155.08 | 159.76 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 174.71 | 162.93 | 162.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 156.47 | 163.10 | 163.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 154.27 | 162.80 | 162.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 162.76 | 162.12 | 162.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 160.75 | 162.30 | 162.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 160.75 | 162.30 | 162.65 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-06 10:15:00 | 158.27 | 162.26 | 162.62 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 133.45 | 124.69 | 132.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 144.11 | 132.19 | 132.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 146.37 | 133.24 | 132.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 138.80 | 141.11 | 137.54 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 127.36 | 135.94 | 135.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 142.87 | 136.01 | 136.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 15:15:00 | 143.70 | 136.23 | 136.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 137.79 | 139.60 | 138.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 139.65 | 139.52 | 138.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 138.90 | 139.88 | 138.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 11:15:00 | 138.35 | 139.87 | 138.61 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 124.11 | 137.66 | 137.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 123.95 | 137.52 | 137.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 124.55 | 118.86 | 124.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 117.18 | 121.41 | 124.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 14:15:00 | 123.01 | 118.98 | 121.95 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 132.44 | 123.62 | 123.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 132.75 | 123.97 | 123.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 131.50 | 131.57 | 128.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 14:15:00 | 133.53 | 131.61 | 128.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 129.54 | 133.94 | 130.81 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 122.00 | 131.23 | 131.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 120.56 | 131.04 | 131.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 09:15:00 | 123.20 | 126.68 | 128.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 125.16 | 126.40 | 128.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-01 12:15:00 | 124.24 | 126.37 | 128.22 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 143.59 | 126.15 | 128.01 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 139.50 | 129.78 | 129.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 140.15 | 129.98 | 129.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 116.76 | 131.42 | 131.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 116.47 | 130.36 | 130.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 118.73 | 118.50 | 123.02 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 132.32 | 124.89 | 124.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 133.79 | 125.06 | 124.95 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 09:15:00 | 158.71 | 2024-12-03 14:15:00 | 159.79 | EXIT_EMA400 | -1.08 |
| SELL | 2025-01-06 09:15:00 | 160.75 | 2025-01-06 14:15:00 | 155.06 | TARGET | 5.69 |
| SELL | 2025-01-06 10:15:00 | 158.27 | 2025-01-13 09:15:00 | 145.21 | TARGET | 13.06 |
| BUY | 2025-07-15 09:15:00 | 139.65 | 2025-07-18 11:15:00 | 144.13 | TARGET | 4.48 |
| SELL | 2025-09-26 09:15:00 | 117.18 | 2025-10-10 14:15:00 | 123.01 | EXIT_EMA400 | -5.83 |
| BUY | 2025-11-26 14:15:00 | 133.53 | 2025-11-28 14:15:00 | 147.95 | TARGET | 14.42 |
| SELL | 2026-01-29 09:15:00 | 123.20 | 2026-02-03 09:15:00 | 143.59 | EXIT_EMA400 | -20.39 |
| SELL | 2026-02-01 12:15:00 | 124.24 | 2026-02-03 09:15:00 | 143.59 | EXIT_EMA400 | -19.35 |
