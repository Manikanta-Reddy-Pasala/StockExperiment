# Adani Power Ltd. (ADANIPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 221.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -1.42
- **Avg P&L per closed trade:** -0.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 128.00 | 138.34 | 138.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 127.26 | 138.23 | 138.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 134.27 | 132.62 | 134.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 09:15:00 | 128.78 | 132.56 | 134.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 133.68 | 132.32 | 134.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-23 15:15:00 | 134.40 | 132.38 | 134.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 15:15:00 | 109.11 | 103.00 | 102.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 10:15:00 | 109.96 | 103.12 | 103.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 106.10 | 107.15 | 105.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 10:15:00 | 112.08 | 107.12 | 105.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-07 09:15:00 | 105.24 | 107.41 | 105.74 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 140.20 | 145.13 | 145.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 149.48 | 145.16 | 145.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 140.26 | 145.14 | 145.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 140.12 | 144.26 | 144.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 142.18 | 140.32 | 142.33 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 148.74 | 143.98 | 143.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 149.76 | 144.03 | 143.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 143.39 | 144.62 | 144.30 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 140.60 | 144.01 | 144.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 136.37 | 143.33 | 143.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 09:15:00 | 143.18 | 141.64 | 142.63 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 155.75 | 143.55 | 143.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 156.40 | 145.28 | 144.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 11:15:00 | 145.25 | 145.64 | 144.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-23 13:15:00 | 148.35 | 145.67 | 144.67 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 09:15:00 | 128.78 | 2024-09-23 15:15:00 | 134.40 | EXIT_EMA400 | -5.62 |
| BUY | 2025-05-05 10:15:00 | 112.08 | 2025-05-07 09:15:00 | 105.24 | EXIT_EMA400 | -6.84 |
| BUY | 2026-03-23 13:15:00 | 148.35 | 2026-04-02 14:15:00 | 159.39 | TARGET | 11.04 |
