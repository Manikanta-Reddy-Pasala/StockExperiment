# Indian Energy Exchange Ltd. (IEX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 125.19
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 13.16
- **Avg P&L per closed trade:** 2.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 13:15:00 | 134.10 | 131.68 | 131.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 10:15:00 | 134.90 | 132.18 | 131.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 132.20 | 133.06 | 132.50 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 123.20 | 131.96 | 131.99 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 138.35 | 131.43 | 131.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 139.10 | 131.76 | 131.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 144.05 | 145.69 | 140.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 148.20 | 145.77 | 140.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 151.20 | 157.82 | 150.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-17 14:15:00 | 147.60 | 157.65 | 150.87 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 10:15:00 | 146.00 | 147.18 | 147.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 12:15:00 | 145.35 | 147.15 | 147.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 147.00 | 146.92 | 147.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-27 14:15:00 | 145.05 | 146.93 | 147.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-04 09:15:00 | 157.00 | 146.19 | 146.64 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 149.00 | 144.49 | 144.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 151.70 | 144.60 | 144.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 149.45 | 149.88 | 147.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-17 14:15:00 | 151.20 | 148.35 | 147.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 153.20 | 152.79 | 150.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 144.20 | 152.70 | 150.23 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 180.08 | 197.19 | 197.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 178.10 | 195.98 | 196.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 11:15:00 | 175.40 | 175.38 | 182.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 14:15:00 | 173.31 | 180.30 | 182.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 10:15:00 | 177.51 | 172.69 | 176.53 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 174.16 | 170.83 | 170.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 175.34 | 170.87 | 170.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 193.54 | 197.77 | 191.18 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 143.55 | 191.32 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 140.38 | 190.36 | 191.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 147.34 | 146.59 | 156.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 144.08 | 147.12 | 155.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-24 11:15:00 | 148.15 | 140.97 | 146.98 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-22 09:15:00 | 148.20 | 2023-12-29 09:15:00 | 169.91 | TARGET | 21.71 |
| SELL | 2024-02-27 14:15:00 | 145.05 | 2024-03-04 09:15:00 | 157.00 | EXIT_EMA400 | -11.95 |
| BUY | 2024-05-17 14:15:00 | 151.20 | 2024-05-21 09:15:00 | 162.87 | TARGET | 11.67 |
| SELL | 2025-01-06 14:15:00 | 173.31 | 2025-01-29 10:15:00 | 177.51 | EXIT_EMA400 | -4.20 |
| SELL | 2025-09-23 10:15:00 | 144.08 | 2025-10-24 11:15:00 | 148.15 | EXIT_EMA400 | -4.07 |
