# CESC Ltd. (CESC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 187.53
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -35.12
- **Avg P&L per closed trade:** -5.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 114.35 | 125.03 | 125.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 114.05 | 124.92 | 124.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 11:15:00 | 123.60 | 123.22 | 124.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-27 14:15:00 | 122.05 | 123.21 | 124.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 122.05 | 123.21 | 124.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-28 09:15:00 | 125.65 | 123.23 | 124.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 137.40 | 124.70 | 124.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 140.65 | 125.92 | 125.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 140.30 | 140.48 | 135.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-17 13:15:00 | 146.60 | 140.72 | 136.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 134.65 | 143.67 | 139.41 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 173.54 | 185.26 | 185.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 170.65 | 185.11 | 185.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 184.80 | 179.72 | 182.04 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 194.65 | 183.95 | 183.90 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 174.69 | 184.64 | 184.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 172.16 | 183.92 | 184.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 138.30 | 137.95 | 149.15 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 13:15:00 | 161.18 | 149.88 | 149.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 14:15:00 | 162.03 | 150.00 | 149.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 163.25 | 163.98 | 159.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 11:15:00 | 167.10 | 163.81 | 159.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 163.13 | 165.74 | 162.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-18 09:15:00 | 164.85 | 165.71 | 162.15 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 161.84 | 165.49 | 162.21 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 164.44 | 168.86 | 168.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 11:15:00 | 164.07 | 168.82 | 168.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 168.60 | 168.08 | 168.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 14:15:00 | 161.61 | 167.97 | 168.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 167.55 | 161.77 | 164.40 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 170.73 | 165.55 | 165.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 172.00 | 165.82 | 165.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 173.24 | 174.33 | 171.06 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 165.65 | 170.79 | 170.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 163.58 | 169.39 | 169.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 13:15:00 | 154.74 | 154.71 | 160.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 09:15:00 | 153.50 | 154.71 | 159.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 158.86 | 154.53 | 158.20 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 171.49 | 157.21 | 157.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 173.01 | 157.36 | 157.23 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-27 14:15:00 | 122.05 | 2024-03-28 09:15:00 | 125.65 | EXIT_EMA400 | -3.60 |
| BUY | 2024-05-17 13:15:00 | 146.60 | 2024-06-04 10:15:00 | 134.65 | EXIT_EMA400 | -11.95 |
| BUY | 2025-06-04 11:15:00 | 167.10 | 2025-06-19 12:15:00 | 161.84 | EXIT_EMA400 | -5.26 |
| BUY | 2025-06-18 09:15:00 | 164.85 | 2025-06-19 12:15:00 | 161.84 | EXIT_EMA400 | -3.01 |
| SELL | 2025-08-26 14:15:00 | 161.61 | 2025-09-15 09:15:00 | 167.55 | EXIT_EMA400 | -5.94 |
| SELL | 2026-02-11 09:15:00 | 153.50 | 2026-02-25 09:15:00 | 158.86 | EXIT_EMA400 | -5.36 |
