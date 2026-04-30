# L&T Finance Ltd. (LTF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 280.80
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 28.02
- **Avg P&L per closed trade:** 9.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 170.25 | 170.82 | 170.82 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 171.15 | 170.82 | 170.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 172.32 | 170.84 | 170.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 170.60 | 170.88 | 170.85 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 170.48 | 170.82 | 170.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 169.45 | 170.78 | 170.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 12:15:00 | 171.09 | 170.76 | 170.79 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 173.40 | 170.83 | 170.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 173.68 | 170.86 | 170.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 170.00 | 170.91 | 170.86 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 15:15:00 | 169.22 | 170.82 | 170.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 166.85 | 170.77 | 170.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 171.83 | 170.08 | 170.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-11 13:15:00 | 166.80 | 170.10 | 170.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-12 09:15:00 | 171.17 | 170.04 | 170.40 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 177.31 | 170.76 | 170.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 179.28 | 172.14 | 171.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 176.59 | 178.12 | 175.15 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 165.60 | 173.14 | 173.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 164.28 | 173.05 | 173.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 147.40 | 146.20 | 153.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 143.88 | 147.12 | 152.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 144.60 | 140.77 | 144.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 147.13 | 140.87 | 144.81 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 158.80 | 142.93 | 142.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 161.75 | 149.94 | 147.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 160.74 | 161.56 | 155.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 167.70 | 161.62 | 155.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-11 11:15:00 | 194.71 | 201.63 | 195.00 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 273.00 | 290.92 | 290.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 271.40 | 290.73 | 290.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 280.07 | 278.49 | 278.48 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-11 13:15:00 | 166.80 | 2024-09-12 09:15:00 | 171.17 | EXIT_EMA400 | -4.37 |
| SELL | 2024-12-13 09:15:00 | 143.88 | 2025-01-21 09:15:00 | 147.13 | EXIT_EMA400 | -3.25 |
| BUY | 2025-05-12 09:15:00 | 167.70 | 2025-06-25 09:15:00 | 203.34 | TARGET | 35.64 |
