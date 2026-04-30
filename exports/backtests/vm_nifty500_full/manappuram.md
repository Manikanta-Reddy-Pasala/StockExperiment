# Manappuram Finance Ltd. (MANAPPURAM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 294.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -17.94
- **Avg P&L per closed trade:** -3.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 13:15:00 | 137.70 | 139.56 | 139.56 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 15:15:00 | 144.80 | 139.59 | 139.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 152.60 | 139.72 | 139.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 167.30 | 168.75 | 161.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 173.30 | 168.75 | 161.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 176.85 | 177.80 | 171.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-14 10:15:00 | 177.60 | 177.80 | 171.17 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 174.65 | 180.12 | 174.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-28 12:15:00 | 172.80 | 180.05 | 174.55 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 169.05 | 181.45 | 181.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 168.60 | 181.32 | 181.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 13:15:00 | 177.90 | 177.57 | 179.35 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 192.04 | 180.62 | 180.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 192.70 | 180.93 | 180.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 204.77 | 206.03 | 197.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 209.49 | 206.00 | 197.33 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 199.38 | 208.37 | 200.82 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 189.67 | 204.73 | 204.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 187.99 | 204.41 | 204.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 163.08 | 159.51 | 170.16 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 188.17 | 174.47 | 174.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 189.67 | 176.01 | 175.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 177.80 | 178.94 | 176.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-08 14:15:00 | 180.03 | 178.95 | 177.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 176.95 | 179.20 | 177.23 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 267.50 | 294.42 | 294.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 264.80 | 293.85 | 294.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 269.50 | 266.36 | 275.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 262.60 | 266.39 | 275.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 274.20 | 267.01 | 273.94 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-11 09:15:00 | 173.30 | 2024-02-28 12:15:00 | 172.80 | EXIT_EMA400 | -0.50 |
| BUY | 2024-02-14 10:15:00 | 177.60 | 2024-02-28 12:15:00 | 172.80 | EXIT_EMA400 | -4.80 |
| BUY | 2024-07-24 09:15:00 | 209.49 | 2024-08-05 10:15:00 | 199.38 | EXIT_EMA400 | -10.11 |
| BUY | 2025-01-08 14:15:00 | 180.03 | 2025-01-09 09:15:00 | 189.10 | TARGET | 9.07 |
| SELL | 2026-04-09 13:15:00 | 262.60 | 2026-04-21 09:15:00 | 274.20 | EXIT_EMA400 | -11.60 |
