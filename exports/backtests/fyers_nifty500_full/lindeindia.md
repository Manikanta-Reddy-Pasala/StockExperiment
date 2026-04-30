# Linde India Ltd. (LINDEINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7350.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 112.63
- **Avg P&L per closed trade:** 22.53

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 12:15:00 | 8573.70 | 7944.15 | 7943.79 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 7524.15 | 8022.62 | 8024.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 7412.05 | 7909.33 | 7960.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 14:15:00 | 6607.95 | 6537.32 | 6906.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 12:15:00 | 6191.65 | 6516.09 | 6839.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6265.00 | 6137.63 | 6505.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 09:15:00 | 6020.85 | 6185.42 | 6436.55 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-20 09:15:00 | 6356.60 | 5989.97 | 6265.60 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 6381.00 | 6220.76 | 6220.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 6401.50 | 6225.28 | 6222.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 6199.00 | 6272.67 | 6249.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 6356.00 | 6241.19 | 6235.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6356.00 | 6241.19 | 6235.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 10:15:00 | 6381.00 | 6242.58 | 6235.89 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 11:15:00 | 6839.00 | 7129.09 | 6882.74 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 6639.00 | 6773.76 | 6773.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 6606.00 | 6769.01 | 6771.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 6559.00 | 6466.98 | 6573.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 12:15:00 | 6350.50 | 6462.75 | 6567.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-09 09:15:00 | 6535.00 | 6442.25 | 6530.96 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 6606.00 | 6004.07 | 6003.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 13:15:00 | 6909.00 | 6105.42 | 6056.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 10:15:00 | 6597.50 | 6769.09 | 6532.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-23 13:15:00 | 6823.50 | 6767.35 | 6535.22 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-13 12:15:00 | 6191.65 | 2025-02-20 09:15:00 | 6356.60 | EXIT_EMA400 | -164.95 |
| SELL | 2025-02-10 09:15:00 | 6020.85 | 2025-02-20 09:15:00 | 6356.60 | EXIT_EMA400 | -335.75 |
| BUY | 2025-05-12 09:15:00 | 6356.00 | 2025-05-14 10:15:00 | 6718.50 | TARGET | 362.50 |
| BUY | 2025-05-12 10:15:00 | 6381.00 | 2025-05-14 12:15:00 | 6816.32 | TARGET | 435.32 |
| SELL | 2025-08-28 12:15:00 | 6350.50 | 2025-09-09 09:15:00 | 6535.00 | EXIT_EMA400 | -184.50 |
