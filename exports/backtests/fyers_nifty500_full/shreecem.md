# Shree Cement Ltd. (SHREECEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 24290.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 1415.53
- **Avg P&L per closed trade:** 202.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 24280.00 | 26778.18 | 26781.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 24100.25 | 25325.32 | 25605.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 25150.00 | 25089.25 | 25437.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 24525.90 | 25047.29 | 25309.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 25545.45 | 24691.35 | 25032.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 27371.80 | 25277.73 | 25272.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 27450.00 | 25318.78 | 25293.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 26582.80 | 26603.86 | 26097.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 26871.10 | 26400.57 | 26081.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 26140.00 | 26392.93 | 26085.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-03 15:15:00 | 26200.00 | 26391.01 | 26085.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 26187.20 | 26388.98 | 26086.35 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 11:15:00 | 26074.05 | 26383.20 | 26086.45 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 25238.05 | 25904.89 | 25905.24 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 26790.75 | 25888.60 | 25887.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 27122.35 | 25927.67 | 25907.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 27273.15 | 27634.04 | 27039.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 14:15:00 | 28176.35 | 27611.79 | 27099.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 29140.00 | 29949.62 | 29133.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-02 12:15:00 | 29085.00 | 29941.02 | 29132.94 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 29915.00 | 30442.66 | 30443.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 29710.00 | 30352.73 | 30396.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 30350.00 | 30333.99 | 30385.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-10 09:15:00 | 30075.00 | 30336.94 | 30384.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 30075.00 | 30336.94 | 30384.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-10 10:15:00 | 29955.00 | 30333.14 | 30381.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 29900.00 | 29900.59 | 30102.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-01 09:15:00 | 29085.00 | 29852.83 | 30069.44 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-15 12:15:00 | 29920.00 | 29657.05 | 29890.24 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 10:15:00 | 24525.90 | 2024-11-25 09:15:00 | 25545.45 | EXIT_EMA400 | -1019.55 |
| BUY | 2025-01-03 09:15:00 | 26871.10 | 2025-01-06 11:15:00 | 26074.05 | EXIT_EMA400 | -797.05 |
| BUY | 2025-01-03 15:15:00 | 26200.00 | 2025-01-06 11:15:00 | 26074.05 | EXIT_EMA400 | -125.95 |
| BUY | 2025-03-05 14:15:00 | 28176.35 | 2025-04-21 10:15:00 | 31407.33 | TARGET | 3230.98 |
| SELL | 2025-09-10 09:15:00 | 30075.00 | 2025-09-26 09:15:00 | 29147.90 | TARGET | 927.10 |
| SELL | 2025-09-10 10:15:00 | 29955.00 | 2025-10-15 12:15:00 | 29920.00 | EXIT_EMA400 | 35.00 |
| SELL | 2025-10-01 09:15:00 | 29085.00 | 2025-10-15 12:15:00 | 29920.00 | EXIT_EMA400 | -835.00 |
