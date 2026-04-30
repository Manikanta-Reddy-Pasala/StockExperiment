# L&T Finance Ltd. (LTF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-04-23 09:15:00 → 2026-04-30 15:30:00 (3478 bars)
- **Last close:** 279.73
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 32.36
- **Avg P&L per closed trade:** 16.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 13:15:00 | 164.35 | 172.74 | 172.75 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 15:15:00 | 176.29 | 171.74 | 171.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 178.29 | 171.96 | 171.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 176.59 | 178.14 | 175.49 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 164.78 | 173.55 | 173.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 164.28 | 173.30 | 173.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 147.40 | 146.26 | 153.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 143.80 | 147.15 | 152.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 144.70 | 140.78 | 144.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 147.07 | 140.87 | 144.86 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 158.80 | 142.90 | 142.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 161.75 | 149.94 | 147.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 160.74 | 161.56 | 155.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 167.70 | 161.62 | 155.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-11 11:15:00 | 194.82 | 201.63 | 195.00 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 271.40 | 290.92 | 291.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 261.75 | 289.22 | 290.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 274.88 | 263.97 | 273.59 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 09:15:00 | 143.80 | 2025-01-21 09:15:00 | 147.07 | EXIT_EMA400 | -3.27 |
| BUY | 2025-05-12 09:15:00 | 167.70 | 2025-06-25 09:15:00 | 203.33 | TARGET | 35.63 |
