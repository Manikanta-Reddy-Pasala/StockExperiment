# Hero MotoCorp Ltd. (HEROMOTOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 5110.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 860.99
- **Avg P&L per closed trade:** 215.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 5260.00 | 5319.34 | 5319.53 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 09:15:00 | 5384.00 | 5319.68 | 5319.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 10:15:00 | 5417.00 | 5320.65 | 5320.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 5289.60 | 5329.15 | 5324.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-30 09:15:00 | 5481.05 | 5330.01 | 5325.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-04 09:15:00 | 5587.45 | 5758.33 | 5618.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 5211.00 | 5539.04 | 5540.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 5165.35 | 5535.32 | 5538.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.11 | 4409.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 4143.95 | 4224.49 | 4376.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 11:15:00 | 3857.50 | 3710.95 | 3842.28 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 4314.90 | 3887.56 | 3887.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 4354.60 | 3892.20 | 3889.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 4256.90 | 4257.48 | 4149.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 4299.00 | 4257.79 | 4150.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.07 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 5379.00 | 5703.60 | 5705.16 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 5754.50 | 5699.06 | 5698.99 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5668.00 | 5698.75 | 5698.83 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 5744.00 | 5699.29 | 5699.10 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 5682.00 | 5698.82 | 5698.87 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 5778.00 | 5699.61 | 5699.27 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 5614.00 | 5698.86 | 5698.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 5590.00 | 5696.43 | 5697.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 5477.50 | 5637.56 | 5659.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5638.93 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-30 09:15:00 | 5481.05 | 2024-09-17 11:15:00 | 5948.00 | TARGET | 466.95 |
| SELL | 2025-02-11 09:15:00 | 4143.95 | 2025-04-07 09:15:00 | 3446.40 | TARGET | 697.55 |
| BUY | 2025-06-24 09:15:00 | 4299.00 | 2025-07-11 15:15:00 | 4203.00 | EXIT_EMA400 | -96.00 |
| SELL | 2026-03-04 09:15:00 | 5477.50 | 2026-03-10 10:15:00 | 5685.00 | EXIT_EMA400 | -207.50 |
