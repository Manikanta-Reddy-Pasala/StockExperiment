# Tata Steel Ltd. (TATASTEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4997 bars)
- **Last close:** 211.36
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 14.97
- **Avg P&L per closed trade:** 3.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 119.35 | 122.51 | 122.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 10:15:00 | 118.90 | 122.23 | 122.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 123.25 | 121.72 | 122.08 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 15:15:00 | 124.35 | 122.39 | 122.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 125.80 | 122.42 | 122.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 128.90 | 129.29 | 126.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 09:15:00 | 130.30 | 129.30 | 126.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 129.55 | 133.64 | 130.94 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 163.76 | 167.92 | 167.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 163.22 | 167.87 | 167.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 154.60 | 153.65 | 157.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-17 11:15:00 | 152.42 | 153.66 | 157.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-24 09:15:00 | 159.06 | 153.22 | 156.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 164.35 | 158.70 | 158.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 155.10 | 158.78 | 158.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 154.25 | 158.73 | 158.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 153.47 | 153.33 | 155.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 12:15:00 | 151.92 | 153.38 | 155.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 149.15 | 146.87 | 149.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-10 09:15:00 | 151.48 | 146.97 | 149.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 154.60 | 142.54 | 140.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 138.53 | 142.39 | 142.40 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 145.05 | 142.33 | 142.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.71 | 142.47 | 142.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.90 | 155.32 | 151.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 156.80 | 154.09 | 151.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 15:15:00 | 157.30 | 160.10 | 157.45 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 160.68 | 170.33 | 170.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.95 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 180.74 | 170.67 | 170.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 200.54 | 201.77 | 192.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-15 09:15:00 | 209.45 | 197.82 | 194.86 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 09:15:00 | 130.30 | 2023-12-29 13:15:00 | 140.87 | TARGET | 10.57 |
| SELL | 2024-09-17 11:15:00 | 152.42 | 2024-09-24 09:15:00 | 159.06 | EXIT_EMA400 | -6.64 |
| SELL | 2024-11-07 12:15:00 | 151.92 | 2024-11-13 09:15:00 | 141.38 | TARGET | 10.54 |
| BUY | 2025-06-24 11:15:00 | 156.80 | 2025-07-31 15:15:00 | 157.30 | EXIT_EMA400 | 0.50 |
