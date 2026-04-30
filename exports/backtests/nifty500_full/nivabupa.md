# Niva Bupa Health Insurance Company Ltd. (NIVABUPA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-14 09:15:00 → 2026-04-30 15:30:00 (2498 bars)
- **Last close:** 78.32
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 1.29
- **Avg P&L per closed trade:** 0.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 75.22 | 79.66 | 79.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 73.50 | 78.55 | 79.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 75.00 | 74.35 | 76.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 13:15:00 | 72.47 | 74.33 | 76.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 77.19 | 74.05 | 75.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 84.40 | 76.27 | 76.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 92.08 | 79.18 | 77.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 83.57 | 86.05 | 82.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 12:15:00 | 85.35 | 82.67 | 82.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 84.91 | 86.16 | 84.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 13:15:00 | 84.33 | 86.14 | 84.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 80.29 | 83.79 | 83.81 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 86.87 | 83.81 | 83.80 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 81.27 | 83.79 | 83.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 80.71 | 82.54 | 83.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 83.48 | 82.36 | 82.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-06 12:15:00 | 80.10 | 82.18 | 82.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-16 09:15:00 | 77.28 | 75.44 | 76.85 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 77.89 | 76.83 | 76.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 79.01 | 76.85 | 76.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 77.11 | 77.30 | 77.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-06 10:15:00 | 77.80 | 77.31 | 77.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-12 09:15:00 | 76.94 | 77.56 | 77.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 15:15:00 | 76.50 | 77.03 | 77.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 76.30 | 77.00 | 77.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 75.76 | 74.45 | 75.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-16 12:15:00 | 70.98 | 74.25 | 75.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 73.70 | 72.31 | 73.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 73.98 | 72.42 | 73.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 79.04 | 74.68 | 74.66 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-25 13:15:00 | 72.47 | 2025-03-27 14:15:00 | 77.19 | EXIT_EMA400 | -4.72 |
| BUY | 2025-07-04 12:15:00 | 85.35 | 2025-07-25 13:15:00 | 84.33 | EXIT_EMA400 | -1.02 |
| SELL | 2025-10-06 12:15:00 | 80.10 | 2025-11-03 09:15:00 | 72.20 | TARGET | 7.90 |
| BUY | 2026-02-06 10:15:00 | 77.80 | 2026-02-09 10:15:00 | 79.93 | TARGET | 2.13 |
| SELL | 2026-03-16 12:15:00 | 70.98 | 2026-04-08 09:15:00 | 73.98 | EXIT_EMA400 | -3.00 |
