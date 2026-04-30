# GAIL (India) Ltd. (GAIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 163.23
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 19.34
- **Avg P&L per closed trade:** 2.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 211.56 | 224.79 | 224.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 210.71 | 224.65 | 224.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 204.95 | 201.68 | 208.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 10:15:00 | 201.23 | 203.95 | 208.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 172.69 | 163.44 | 171.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 13:15:00 | 195.65 | 174.89 | 174.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 196.06 | 187.15 | 183.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 187.59 | 188.65 | 184.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 10:15:00 | 193.06 | 189.06 | 185.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 189.03 | 190.54 | 186.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 11:15:00 | 189.79 | 190.52 | 186.76 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-18 11:15:00 | 187.03 | 190.48 | 187.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 179.90 | 186.55 | 186.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 177.96 | 185.79 | 186.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 179.43 | 176.83 | 179.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 10:15:00 | 175.06 | 177.09 | 179.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 178.70 | 176.11 | 178.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-11 12:15:00 | 179.05 | 176.18 | 178.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 184.00 | 178.84 | 178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 184.29 | 179.17 | 179.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-07 11:15:00 | 180.08 | 179.67 | 179.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 180.86 | 181.60 | 180.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-25 10:15:00 | 182.10 | 181.60 | 180.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 180.62 | 181.58 | 180.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-25 13:15:00 | 180.07 | 181.57 | 180.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 171.10 | 179.87 | 179.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 170.44 | 179.69 | 179.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 172.89 | 172.69 | 174.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 10:15:00 | 170.97 | 172.86 | 174.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 169.54 | 167.16 | 170.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-29 11:15:00 | 166.75 | 167.18 | 170.55 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 166.15 | 164.81 | 167.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-19 09:15:00 | 168.40 | 164.98 | 167.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 10:15:00 | 201.23 | 2025-01-10 09:15:00 | 179.60 | TARGET | 21.63 |
| BUY | 2025-06-09 10:15:00 | 193.06 | 2025-06-18 11:15:00 | 187.03 | EXIT_EMA400 | -6.03 |
| BUY | 2025-06-13 11:15:00 | 189.79 | 2025-06-18 11:15:00 | 187.03 | EXIT_EMA400 | -2.76 |
| SELL | 2025-09-04 10:15:00 | 175.06 | 2025-09-11 12:15:00 | 179.05 | EXIT_EMA400 | -3.99 |
| BUY | 2025-11-07 11:15:00 | 180.08 | 2025-11-11 11:15:00 | 182.43 | TARGET | 2.35 |
| BUY | 2025-11-25 10:15:00 | 182.10 | 2025-11-25 13:15:00 | 180.07 | EXIT_EMA400 | -2.03 |
| SELL | 2026-01-06 10:15:00 | 170.97 | 2026-01-27 09:15:00 | 159.15 | TARGET | 11.82 |
| SELL | 2026-01-29 11:15:00 | 166.75 | 2026-02-19 09:15:00 | 168.40 | EXIT_EMA400 | -1.65 |
