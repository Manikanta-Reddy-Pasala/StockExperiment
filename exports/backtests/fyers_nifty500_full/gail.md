# GAIL (India) Ltd. (GAIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 163.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -5.20
- **Avg P&L per closed trade:** -0.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 211.55 | 224.78 | 224.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 210.71 | 224.64 | 224.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 214.50 | 214.49 | 218.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 09:15:00 | 206.90 | 214.30 | 218.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 207.76 | 201.90 | 208.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 10:15:00 | 208.96 | 201.97 | 208.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 13:15:00 | 195.66 | 174.88 | 174.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 196.06 | 187.15 | 183.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 187.40 | 188.65 | 184.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 10:15:00 | 193.06 | 189.06 | 185.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 189.03 | 190.53 | 186.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 14:15:00 | 191.36 | 190.50 | 186.78 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-18 11:15:00 | 187.03 | 190.46 | 187.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 179.90 | 186.55 | 186.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 177.90 | 185.79 | 186.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 10:15:00 | 175.06 | 177.08 | 179.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 178.70 | 176.12 | 178.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-11 12:15:00 | 179.00 | 176.19 | 178.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 184.00 | 178.84 | 178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 184.25 | 179.17 | 179.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-07 11:15:00 | 180.08 | 179.68 | 179.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 180.86 | 181.60 | 180.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-25 10:15:00 | 182.10 | 181.60 | 180.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 180.62 | 181.59 | 180.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-25 13:15:00 | 180.07 | 181.57 | 180.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 171.09 | 179.88 | 179.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 170.44 | 179.70 | 179.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 172.89 | 172.69 | 174.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 10:15:00 | 170.97 | 172.87 | 174.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 169.60 | 167.16 | 170.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-29 11:15:00 | 166.75 | 167.18 | 170.55 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 166.15 | 164.71 | 167.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-18 13:15:00 | 167.73 | 164.80 | 167.63 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 09:15:00 | 206.90 | 2024-12-05 10:15:00 | 208.96 | EXIT_EMA400 | -2.06 |
| BUY | 2025-06-09 10:15:00 | 193.06 | 2025-06-18 11:15:00 | 187.03 | EXIT_EMA400 | -6.03 |
| BUY | 2025-06-13 14:15:00 | 191.36 | 2025-06-18 11:15:00 | 187.03 | EXIT_EMA400 | -4.33 |
| SELL | 2025-09-04 10:15:00 | 175.06 | 2025-09-11 12:15:00 | 179.00 | EXIT_EMA400 | -3.94 |
| BUY | 2025-11-07 11:15:00 | 180.08 | 2025-11-11 11:15:00 | 182.42 | TARGET | 2.34 |
| BUY | 2025-11-25 10:15:00 | 182.10 | 2025-11-25 13:15:00 | 180.07 | EXIT_EMA400 | -2.03 |
| SELL | 2026-01-06 10:15:00 | 170.97 | 2026-01-27 09:15:00 | 159.15 | TARGET | 11.82 |
| SELL | 2026-01-29 11:15:00 | 166.75 | 2026-02-18 13:15:00 | 167.73 | EXIT_EMA400 | -0.98 |
