# City Union Bank Ltd. (CUB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 270.09
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 51.86
- **Avg P&L per closed trade:** 6.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 124.75 | 127.60 | 127.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 124.25 | 127.56 | 127.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 127.20 | 126.56 | 127.03 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 12:15:00 | 133.10 | 127.45 | 127.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 135.95 | 127.69 | 127.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 13:15:00 | 129.50 | 129.54 | 128.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-22 10:15:00 | 130.40 | 129.55 | 128.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-09-25 10:15:00 | 128.65 | 129.57 | 128.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 135.95 | 145.77 | 145.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 133.65 | 145.55 | 145.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 12:15:00 | 140.90 | 140.49 | 142.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-19 09:15:00 | 138.60 | 140.45 | 142.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 137.00 | 134.15 | 137.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 09:15:00 | 138.90 | 134.58 | 137.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 12:15:00 | 153.15 | 139.41 | 139.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 13:15:00 | 153.60 | 139.55 | 139.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 152.30 | 152.82 | 148.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 11:15:00 | 154.55 | 152.14 | 148.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-21 09:15:00 | 145.70 | 152.35 | 148.92 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 154.50 | 163.96 | 164.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 154.22 | 163.77 | 163.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 168.09 | 159.79 | 161.62 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 174.99 | 163.18 | 163.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 176.15 | 163.31 | 163.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 12:15:00 | 169.74 | 170.08 | 167.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-13 13:15:00 | 171.77 | 170.10 | 167.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 168.06 | 170.40 | 167.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-11-21 12:15:00 | 170.63 | 170.39 | 167.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 174.51 | 179.34 | 175.07 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 167.00 | 173.17 | 173.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 164.81 | 172.19 | 172.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 171.60 | 171.28 | 172.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 12:15:00 | 165.19 | 172.37 | 172.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 162.10 | 156.38 | 161.31 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 182.40 | 162.75 | 162.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 184.25 | 162.96 | 162.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 192.86 | 193.07 | 184.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 12:15:00 | 195.25 | 193.10 | 184.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-25 09:15:00 | 206.85 | 211.76 | 206.89 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 247.05 | 274.94 | 274.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 242.25 | 274.61 | 274.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 257.50 | 252.91 | 260.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 11:15:00 | 253.02 | 253.11 | 260.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-17 15:15:00 | 259.22 | 252.99 | 259.18 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-22 10:15:00 | 130.40 | 2023-09-25 10:15:00 | 128.65 | EXIT_EMA400 | -1.75 |
| SELL | 2024-02-19 09:15:00 | 138.60 | 2024-03-13 14:15:00 | 126.44 | TARGET | 12.16 |
| BUY | 2024-05-14 11:15:00 | 154.55 | 2024-05-21 09:15:00 | 145.70 | EXIT_EMA400 | -8.85 |
| BUY | 2024-11-21 12:15:00 | 170.63 | 2024-11-26 14:15:00 | 179.22 | TARGET | 8.59 |
| BUY | 2024-11-13 13:15:00 | 171.77 | 2024-12-04 09:15:00 | 185.35 | TARGET | 13.58 |
| SELL | 2025-02-11 12:15:00 | 165.19 | 2025-03-21 09:15:00 | 162.10 | EXIT_EMA400 | 3.09 |
| BUY | 2025-06-13 12:15:00 | 195.25 | 2025-07-01 13:15:00 | 226.49 | TARGET | 31.24 |
| SELL | 2026-04-09 11:15:00 | 253.02 | 2026-04-17 15:15:00 | 259.22 | EXIT_EMA400 | -6.20 |
