# Mangalore Refinery & Petrochemicals Ltd. (MRPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 168.37
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -24.15
- **Avg P&L per closed trade:** -4.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 205.28 | 214.40 | 214.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 204.00 | 213.89 | 214.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 213.27 | 212.67 | 213.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 09:15:00 | 209.07 | 212.76 | 213.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 123.50 | 116.68 | 124.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 13:15:00 | 129.00 | 116.80 | 124.46 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 140.70 | 129.10 | 129.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 143.30 | 129.24 | 129.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 131.70 | 132.97 | 131.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-15 09:15:00 | 137.22 | 131.94 | 131.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 135.18 | 140.05 | 137.00 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 125.39 | 139.68 | 139.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 124.46 | 138.30 | 139.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 128.85 | 127.85 | 131.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 09:15:00 | 126.89 | 127.88 | 131.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 130.05 | 127.89 | 130.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-18 09:15:00 | 130.80 | 127.92 | 130.71 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 148.61 | 131.94 | 131.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 149.59 | 132.11 | 132.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 160.23 | 163.85 | 154.13 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 146.81 | 152.05 | 152.08 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 157.65 | 152.11 | 152.10 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 150.60 | 152.09 | 152.10 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 153.87 | 152.11 | 152.11 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 151.40 | 152.10 | 152.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 150.32 | 152.08 | 152.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 152.48 | 149.02 | 150.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-19 09:15:00 | 145.75 | 149.40 | 150.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-21 12:15:00 | 153.07 | 148.52 | 149.96 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 170.09 | 151.12 | 151.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 174.43 | 151.35 | 151.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 186.86 | 186.93 | 176.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 10:15:00 | 189.01 | 186.92 | 176.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 178.47 | 186.74 | 177.35 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 09:15:00 | 187.23 | 186.67 | 177.41 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-24 09:15:00 | 176.19 | 188.34 | 180.14 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-27 09:15:00 | 209.07 | 2024-09-09 09:15:00 | 196.09 | TARGET | 12.98 |
| BUY | 2025-05-15 09:15:00 | 137.22 | 2025-06-16 09:15:00 | 135.18 | EXIT_EMA400 | -2.04 |
| SELL | 2025-09-12 09:15:00 | 126.89 | 2025-09-18 09:15:00 | 130.80 | EXIT_EMA400 | -3.91 |
| SELL | 2026-01-19 09:15:00 | 145.75 | 2026-01-21 12:15:00 | 153.07 | EXIT_EMA400 | -7.32 |
| BUY | 2026-03-12 10:15:00 | 189.01 | 2026-03-24 09:15:00 | 176.19 | EXIT_EMA400 | -12.82 |
| BUY | 2026-03-16 09:15:00 | 187.23 | 2026-03-24 09:15:00 | 176.19 | EXIT_EMA400 | -11.04 |
