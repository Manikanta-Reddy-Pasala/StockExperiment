# CESC Ltd. (CESC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 188.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -25.96
- **Avg P&L per closed trade:** -5.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 174.24 | 185.40 | 185.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 173.54 | 185.28 | 185.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 184.80 | 179.74 | 182.10 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 194.65 | 183.96 | 183.95 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 174.69 | 184.64 | 184.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 172.14 | 183.92 | 184.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 138.30 | 137.68 | 148.69 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 11:15:00 | 161.08 | 149.60 | 149.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 162.73 | 153.82 | 152.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 163.25 | 163.97 | 159.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 10:15:00 | 165.75 | 163.78 | 159.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 163.13 | 165.76 | 162.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-18 09:15:00 | 164.85 | 165.72 | 162.11 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 161.84 | 165.50 | 162.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 164.07 | 168.82 | 168.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 163.56 | 168.68 | 168.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 14:15:00 | 161.55 | 167.96 | 168.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 167.60 | 161.76 | 164.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 171.11 | 165.49 | 165.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 171.41 | 165.75 | 165.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 173.24 | 174.31 | 171.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-17 15:15:00 | 175.99 | 173.86 | 171.39 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-19 13:15:00 | 171.36 | 173.80 | 171.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 165.65 | 170.80 | 170.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 163.70 | 169.39 | 169.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 12:15:00 | 154.66 | 154.26 | 159.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 150.50 | 154.29 | 159.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 158.86 | 154.33 | 157.91 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 171.49 | 157.03 | 156.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 173.40 | 157.34 | 157.14 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-04 10:15:00 | 165.75 | 2025-06-19 12:15:00 | 161.84 | EXIT_EMA400 | -3.91 |
| BUY | 2025-06-18 09:15:00 | 164.85 | 2025-06-19 12:15:00 | 161.84 | EXIT_EMA400 | -3.01 |
| SELL | 2025-08-26 14:15:00 | 161.55 | 2025-09-15 09:15:00 | 167.60 | EXIT_EMA400 | -6.05 |
| BUY | 2025-11-17 15:15:00 | 175.99 | 2025-11-19 13:15:00 | 171.36 | EXIT_EMA400 | -4.63 |
| SELL | 2026-02-13 09:15:00 | 150.50 | 2026-02-25 09:15:00 | 158.86 | EXIT_EMA400 | -8.36 |
