# Divi's Laboratories Ltd. (DIVISLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 6508.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -24.86
- **Avg P&L per closed trade:** -4.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 5666.35 | 5820.58 | 5820.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 5610.45 | 5814.96 | 5817.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.64 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 6150.00 | 5820.92 | 5819.29 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 5435.00 | 5841.66 | 5842.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5373.15 | 5837.00 | 5839.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 5737.25 | 5707.01 | 5763.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 14:15:00 | 5619.85 | 5705.38 | 5761.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 5768.00 | 5703.16 | 5756.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.83 | 5750.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.43 | 5758.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.42 | 5839.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 11:15:00 | 5948.50 | 5905.85 | 5840.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6581.50 | 6694.16 | 6568.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-29 09:15:00 | 6534.00 | 6692.56 | 6568.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 5980.00 | 6491.92 | 6494.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 5959.50 | 6462.34 | 6479.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-22 09:15:00 | 6139.00 | 6149.55 | 6246.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-08 09:15:00 | 6186.00 | 5999.56 | 6127.59 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6228.92 | 6222.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.72 | 6400.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 6565.50 | 6416.35 | 6401.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 6425.50 | 6439.16 | 6416.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-29 09:15:00 | 6390.50 | 6438.59 | 6416.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6407.30 | 6410.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6241.50 | 6230.99 | 6307.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 10:15:00 | 6073.00 | 6228.86 | 6303.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.28 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6294.15 | 6294.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6292.73 | 6293.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 11:15:00 | 6091.00 | 6061.97 | 6149.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 6031.50 | 6063.78 | 6148.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6127.50 | 6064.95 | 6145.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 12:15:00 | 6168.00 | 6067.12 | 6145.77 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 6444.50 | 6198.32 | 6198.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.03 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-13 14:15:00 | 5619.85 | 2025-03-18 14:15:00 | 5768.00 | EXIT_EMA400 | -148.15 |
| BUY | 2025-05-09 11:15:00 | 5948.50 | 2025-05-14 13:15:00 | 6273.60 | TARGET | 325.10 |
| SELL | 2025-09-22 09:15:00 | 6139.00 | 2025-09-26 09:15:00 | 5815.81 | TARGET | 323.19 |
| BUY | 2025-12-19 09:15:00 | 6565.50 | 2025-12-29 09:15:00 | 6390.50 | EXIT_EMA400 | -175.00 |
| SELL | 2026-02-04 10:15:00 | 6073.00 | 2026-02-11 11:15:00 | 6286.50 | EXIT_EMA400 | -213.50 |
| SELL | 2026-04-13 09:15:00 | 6031.50 | 2026-04-15 12:15:00 | 6168.00 | EXIT_EMA400 | -136.50 |
