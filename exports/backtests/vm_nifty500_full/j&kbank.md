# Jammu & Kashmir Bank Ltd. (J&KBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 129.02
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 11.62
- **Avg P&L per closed trade:** 1.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 126.45 | 133.34 | 133.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 124.90 | 130.89 | 131.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 132.00 | 130.75 | 131.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 10:15:00 | 120.35 | 130.92 | 131.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-02 10:15:00 | 118.12 | 112.99 | 117.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 13:15:00 | 114.05 | 96.52 | 96.49 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 92.05 | 96.70 | 96.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 91.25 | 96.41 | 96.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 96.60 | 95.81 | 96.25 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 101.15 | 96.66 | 96.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 103.41 | 96.86 | 96.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 102.99 | 103.33 | 101.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 104.55 | 103.21 | 101.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 106.10 | 110.14 | 107.31 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 103.70 | 105.69 | 105.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 103.20 | 105.62 | 105.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 10:15:00 | 103.45 | 102.69 | 103.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 100.73 | 103.30 | 103.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-30 09:15:00 | 103.74 | 102.98 | 103.66 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 104.65 | 104.04 | 104.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 105.51 | 104.06 | 104.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 105.55 | 105.55 | 104.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 107.09 | 105.32 | 104.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 105.62 | 106.34 | 105.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-21 11:15:00 | 106.05 | 106.34 | 105.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 105.90 | 106.33 | 105.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-21 14:15:00 | 105.09 | 106.31 | 105.57 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 101.37 | 105.26 | 105.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 100.94 | 105.18 | 105.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 102.02 | 101.72 | 103.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 11:15:00 | 101.47 | 102.22 | 103.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-14 12:15:00 | 102.84 | 101.87 | 102.83 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 105.92 | 103.28 | 103.28 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 14:15:00 | 102.61 | 103.28 | 103.29 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 104.70 | 103.30 | 103.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 105.75 | 103.39 | 103.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 103.05 | 103.55 | 103.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-23 09:15:00 | 105.77 | 103.57 | 103.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 10:15:00 | 110.80 | 115.18 | 110.96 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-04 10:15:00 | 120.35 | 2024-08-02 10:15:00 | 118.12 | EXIT_EMA400 | 2.23 |
| BUY | 2025-06-24 09:15:00 | 104.55 | 2025-06-30 09:15:00 | 114.30 | TARGET | 9.75 |
| SELL | 2025-09-26 09:15:00 | 100.73 | 2025-09-30 09:15:00 | 103.74 | EXIT_EMA400 | -3.01 |
| BUY | 2025-11-10 09:15:00 | 107.09 | 2025-11-21 14:15:00 | 105.09 | EXIT_EMA400 | -2.00 |
| BUY | 2025-11-21 11:15:00 | 106.05 | 2025-11-21 14:15:00 | 105.09 | EXIT_EMA400 | -0.96 |
| SELL | 2026-01-09 11:15:00 | 101.47 | 2026-01-14 12:15:00 | 102.84 | EXIT_EMA400 | -1.37 |
| BUY | 2026-02-23 09:15:00 | 105.77 | 2026-02-24 09:15:00 | 112.75 | TARGET | 6.98 |
