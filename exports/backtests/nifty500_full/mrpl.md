# Mangalore Refinery & Petrochemicals Ltd. (MRPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 167.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** -2.16
- **Avg P&L per closed trade:** -0.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 15:15:00 | 199.45 | 215.48 | 215.56 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 216.15 | 215.04 | 215.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 218.08 | 215.11 | 215.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 214.50 | 215.13 | 215.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-03 11:15:00 | 218.82 | 215.14 | 215.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 218.82 | 215.14 | 215.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-05 09:15:00 | 219.90 | 215.49 | 215.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-19 09:15:00 | 216.50 | 224.19 | 220.26 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 11:15:00 | 204.35 | 218.03 | 218.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 202.78 | 216.80 | 217.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 213.27 | 212.90 | 215.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 13:15:00 | 211.01 | 213.00 | 214.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 123.59 | 116.75 | 124.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 13:15:00 | 128.96 | 116.87 | 124.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 10:15:00 | 143.11 | 129.28 | 129.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 144.59 | 135.59 | 133.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 139.75 | 140.13 | 137.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 10:15:00 | 143.92 | 138.81 | 136.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 140.23 | 143.75 | 141.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 125.39 | 139.68 | 139.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 124.46 | 138.30 | 139.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 128.85 | 127.85 | 131.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 09:15:00 | 126.89 | 127.88 | 131.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-18 09:15:00 | 130.80 | 127.92 | 130.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 148.71 | 131.93 | 131.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 149.66 | 132.11 | 132.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 160.23 | 163.85 | 154.13 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 146.81 | 152.06 | 152.08 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 157.65 | 152.11 | 152.11 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 150.32 | 152.10 | 152.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 12:15:00 | 149.98 | 152.06 | 152.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 152.48 | 149.04 | 150.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-19 09:15:00 | 145.72 | 149.41 | 150.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-21 12:15:00 | 153.07 | 148.53 | 149.97 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 174.40 | 151.36 | 151.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 179.35 | 156.15 | 153.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 183.48 | 186.79 | 176.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 10:15:00 | 189.12 | 186.81 | 176.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 178.29 | 186.64 | 177.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 09:15:00 | 187.23 | 186.57 | 177.23 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-24 09:15:00 | 176.31 | 188.27 | 180.00 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-03 11:15:00 | 218.82 | 2024-07-09 09:15:00 | 230.00 | TARGET | 11.18 |
| BUY | 2024-07-05 09:15:00 | 219.90 | 2024-07-09 09:15:00 | 233.77 | TARGET | 13.87 |
| SELL | 2024-08-26 13:15:00 | 211.01 | 2024-09-06 09:15:00 | 199.54 | TARGET | 11.47 |
| BUY | 2025-06-23 10:15:00 | 143.92 | 2025-07-21 09:15:00 | 140.23 | EXIT_EMA400 | -3.69 |
| SELL | 2025-09-12 09:15:00 | 126.89 | 2025-09-18 09:15:00 | 130.80 | EXIT_EMA400 | -3.91 |
| SELL | 2026-01-19 09:15:00 | 145.72 | 2026-01-21 12:15:00 | 153.07 | EXIT_EMA400 | -7.35 |
| BUY | 2026-03-12 10:15:00 | 189.12 | 2026-03-24 09:15:00 | 176.31 | EXIT_EMA400 | -12.81 |
| BUY | 2026-03-16 09:15:00 | 187.23 | 2026-03-24 09:15:00 | 176.31 | EXIT_EMA400 | -10.92 |
