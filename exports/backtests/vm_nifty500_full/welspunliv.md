# Welspun Living Ltd. (WELSPUNLIV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-12-14 09:15:00 → 2026-04-30 15:30:00 (4073 bars)
- **Last close:** 128.96
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 9 |
| ENTRY1 | 9 |
| ENTRY2 | 3 |
| EXIT | 9 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 17.45
- **Avg P&L per closed trade:** 1.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 12:15:00 | 147.50 | 153.31 | 153.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 13:15:00 | 147.15 | 153.25 | 153.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 13:15:00 | 155.90 | 152.88 | 153.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-12 12:15:00 | 147.40 | 152.73 | 153.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 149.30 | 147.31 | 149.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 11:15:00 | 149.75 | 147.33 | 149.74 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 11:15:00 | 160.25 | 151.07 | 151.06 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 147.10 | 151.09 | 151.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 145.55 | 151.03 | 151.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 148.20 | 146.94 | 148.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 09:15:00 | 141.00 | 145.61 | 147.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 140.56 | 140.79 | 144.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-13 10:15:00 | 140.29 | 140.78 | 144.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 143.44 | 140.82 | 143.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-18 09:15:00 | 146.60 | 141.02 | 143.97 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 152.17 | 145.72 | 145.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 161.71 | 147.16 | 146.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 172.62 | 173.85 | 165.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-16 09:15:00 | 174.90 | 173.87 | 165.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 178.85 | 185.58 | 177.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-12 11:15:00 | 181.39 | 185.36 | 177.51 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 177.67 | 185.06 | 177.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-13 11:15:00 | 177.25 | 184.98 | 177.58 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 13:15:00 | 163.31 | 174.43 | 174.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 159.67 | 169.62 | 171.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 12:15:00 | 163.00 | 160.81 | 165.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 09:15:00 | 154.29 | 160.67 | 165.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 159.79 | 155.10 | 159.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 15:15:00 | 160.10 | 155.15 | 159.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 174.68 | 163.06 | 163.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 155.43 | 163.25 | 163.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 154.27 | 162.81 | 163.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 162.76 | 162.12 | 162.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 160.75 | 162.31 | 162.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 160.75 | 162.31 | 162.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-06 10:15:00 | 158.18 | 162.27 | 162.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 133.45 | 124.77 | 132.60 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 144.11 | 132.20 | 132.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 146.37 | 133.24 | 132.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 138.79 | 141.11 | 137.57 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 127.20 | 136.03 | 136.04 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 142.87 | 136.02 | 136.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 15:15:00 | 143.70 | 136.24 | 136.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 137.75 | 139.60 | 138.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 139.65 | 139.52 | 138.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 138.90 | 139.89 | 138.63 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 11:15:00 | 138.35 | 139.87 | 138.63 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 124.11 | 137.66 | 137.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 123.95 | 137.53 | 137.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 124.55 | 118.87 | 124.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 12:15:00 | 120.26 | 119.53 | 124.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 124.18 | 119.72 | 124.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 13:15:00 | 124.44 | 119.89 | 124.32 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 132.40 | 123.62 | 123.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 132.75 | 123.97 | 123.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 131.50 | 131.58 | 128.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 14:15:00 | 133.52 | 131.61 | 128.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 129.54 | 133.93 | 130.81 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 122.00 | 131.24 | 131.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 121.01 | 131.05 | 131.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 127.78 | 126.74 | 128.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 09:15:00 | 123.20 | 126.68 | 128.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 143.59 | 126.26 | 128.13 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 139.50 | 129.87 | 129.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 140.15 | 130.07 | 129.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 132.69 | 135.30 | 133.18 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 116.76 | 131.44 | 131.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 116.46 | 130.38 | 130.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 118.73 | 118.51 | 123.06 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 133.79 | 124.95 | 124.94 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-12 12:15:00 | 147.40 | 2024-04-01 11:15:00 | 149.75 | EXIT_EMA400 | -2.35 |
| SELL | 2024-05-29 09:15:00 | 141.00 | 2024-06-18 09:15:00 | 146.60 | EXIT_EMA400 | -5.60 |
| SELL | 2024-06-13 10:15:00 | 140.29 | 2024-06-18 09:15:00 | 146.60 | EXIT_EMA400 | -6.31 |
| BUY | 2024-08-16 09:15:00 | 174.90 | 2024-08-27 09:15:00 | 202.87 | TARGET | 27.97 |
| BUY | 2024-09-12 11:15:00 | 181.39 | 2024-09-13 11:15:00 | 177.25 | EXIT_EMA400 | -4.14 |
| SELL | 2024-11-11 09:15:00 | 154.29 | 2024-12-03 15:15:00 | 160.10 | EXIT_EMA400 | -5.81 |
| SELL | 2025-01-06 09:15:00 | 160.75 | 2025-01-08 13:15:00 | 154.86 | TARGET | 5.89 |
| SELL | 2025-01-06 10:15:00 | 158.18 | 2025-01-13 09:15:00 | 144.65 | TARGET | 13.53 |
| BUY | 2025-07-15 09:15:00 | 139.65 | 2025-07-18 11:15:00 | 144.09 | TARGET | 4.44 |
| SELL | 2025-09-12 12:15:00 | 120.26 | 2025-09-16 13:15:00 | 124.44 | EXIT_EMA400 | -4.18 |
| BUY | 2025-11-26 14:15:00 | 133.52 | 2025-11-28 14:15:00 | 147.91 | TARGET | 14.39 |
| SELL | 2026-01-29 09:15:00 | 123.20 | 2026-02-03 09:15:00 | 143.59 | EXIT_EMA400 | -20.39 |
