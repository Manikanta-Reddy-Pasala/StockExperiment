# Tata Steel Ltd. (TATASTEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 211.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 22.20
- **Avg P&L per closed trade:** 7.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 164.11 | 158.92 | 158.92 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 154.45 | 158.92 | 158.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 153.81 | 158.64 | 158.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 153.29 | 153.27 | 155.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 14:15:00 | 150.75 | 153.27 | 155.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 149.12 | 146.86 | 149.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-10 09:15:00 | 151.48 | 146.96 | 149.91 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 151.50 | 137.99 | 137.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.49 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 139.60 | 142.36 | 142.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 145.30 | 142.30 | 142.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.74 | 142.47 | 142.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.85 | 155.31 | 151.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 156.79 | 154.08 | 151.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 15:15:00 | 157.30 | 160.09 | 157.44 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 160.68 | 170.33 | 170.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.94 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 181.45 | 171.06 | 170.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 182.79 | 184.32 | 179.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 13:15:00 | 187.02 | 184.35 | 179.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 190.89 | 201.25 | 193.79 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 14:15:00 | 150.75 | 2024-12-10 09:15:00 | 151.48 | EXIT_EMA400 | -0.73 |
| BUY | 2025-06-24 11:15:00 | 156.79 | 2025-07-31 15:15:00 | 157.30 | EXIT_EMA400 | 0.51 |
| BUY | 2026-02-02 13:15:00 | 187.02 | 2026-02-10 10:15:00 | 209.44 | TARGET | 22.42 |
