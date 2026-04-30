# Divi's Laboratories Ltd. (DIVISLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 6502.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 10 |
| ENTRY2 | 1 |
| EXIT | 10 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 360.32
- **Avg P&L per closed trade:** 32.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 3463.25 | 3679.57 | 3679.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 3433.65 | 3677.13 | 3678.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 3542.50 | 3536.04 | 3590.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-13 11:15:00 | 3496.50 | 3535.34 | 3589.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-17 09:15:00 | 3597.00 | 3535.38 | 3584.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 3797.95 | 3619.60 | 3619.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 3816.70 | 3635.97 | 3627.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 11:15:00 | 3666.00 | 3666.15 | 3645.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-15 09:15:00 | 3715.25 | 3666.03 | 3648.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 3660.05 | 3676.73 | 3656.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-20 14:15:00 | 3619.55 | 3676.16 | 3655.84 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 3652.10 | 3727.70 | 3728.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 13:15:00 | 3646.75 | 3726.90 | 3727.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 3771.60 | 3725.87 | 3727.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-14 09:15:00 | 3641.65 | 3725.77 | 3726.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 3721.80 | 3723.65 | 3725.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-16 09:15:00 | 3759.10 | 3723.35 | 3725.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 3812.95 | 3658.14 | 3646.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 3771.10 | 3799.05 | 3733.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-10 09:15:00 | 3820.50 | 3798.88 | 3735.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5828.40 | 5939.02 | 5799.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-20 09:15:00 | 5865.00 | 5932.04 | 5800.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 5828.90 | 5922.21 | 5805.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-24 14:15:00 | 5801.35 | 5920.05 | 5806.03 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 5650.85 | 5819.20 | 5819.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 5610.45 | 5815.25 | 5817.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 5805.15 | 5795.72 | 5807.46 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 6096.25 | 5819.03 | 5818.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 6176.25 | 5822.58 | 5820.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 5863.35 | 5886.07 | 5855.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-11 14:15:00 | 5949.80 | 5886.83 | 5856.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-12 09:15:00 | 5836.50 | 5886.92 | 5856.63 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 5436.05 | 5845.59 | 5845.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5373.15 | 5840.89 | 5843.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 5737.25 | 5709.48 | 5766.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 14:15:00 | 5618.60 | 5707.72 | 5763.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 5767.90 | 5705.24 | 5758.86 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.52 | 5751.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.13 | 5758.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.76 | 5840.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 11:15:00 | 5948.50 | 5906.19 | 5840.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6582.00 | 6694.32 | 6568.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-29 09:15:00 | 6534.00 | 6692.72 | 6568.76 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 5992.00 | 6496.40 | 6496.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 5968.00 | 6486.17 | 6491.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6258.50 | 6145.76 | 6248.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-22 09:15:00 | 6139.00 | 6149.72 | 6246.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-08 09:15:00 | 6186.00 | 5999.70 | 6127.70 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.59 | 6216.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6229.09 | 6222.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.56 | 6400.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 6565.50 | 6416.51 | 6402.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 6425.00 | 6439.41 | 6416.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-29 09:15:00 | 6390.50 | 6438.83 | 6416.67 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6410.41 | 6411.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6406.56 | 6410.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 6290.00 | 6244.55 | 6315.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 12:15:00 | 6224.50 | 6244.35 | 6315.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6198.42 | 6276.47 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 6468.00 | 6201.43 | 6201.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 6519.00 | 6210.13 | 6205.72 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-13 11:15:00 | 3496.50 | 2023-11-17 09:15:00 | 3597.00 | EXIT_EMA400 | -100.50 |
| BUY | 2023-12-15 09:15:00 | 3715.25 | 2023-12-20 14:15:00 | 3619.55 | EXIT_EMA400 | -95.70 |
| SELL | 2024-02-14 09:15:00 | 3641.65 | 2024-02-16 09:15:00 | 3759.10 | EXIT_EMA400 | -117.45 |
| BUY | 2024-05-10 09:15:00 | 3820.50 | 2024-05-23 11:15:00 | 4076.92 | TARGET | 256.42 |
| BUY | 2024-12-20 09:15:00 | 5865.00 | 2024-12-24 14:15:00 | 5801.35 | EXIT_EMA400 | -63.65 |
| BUY | 2025-02-11 14:15:00 | 5949.80 | 2025-02-12 09:15:00 | 5836.50 | EXIT_EMA400 | -113.30 |
| SELL | 2025-03-13 14:15:00 | 5618.60 | 2025-03-18 14:15:00 | 5767.90 | EXIT_EMA400 | -149.30 |
| BUY | 2025-05-09 11:15:00 | 5948.50 | 2025-05-14 13:15:00 | 6271.35 | TARGET | 322.85 |
| SELL | 2025-09-22 09:15:00 | 6139.00 | 2025-09-26 09:15:00 | 5815.49 | TARGET | 323.51 |
| BUY | 2025-12-19 09:15:00 | 6565.50 | 2025-12-29 09:15:00 | 6390.50 | EXIT_EMA400 | -175.00 |
| SELL | 2026-02-03 12:15:00 | 6224.50 | 2026-02-05 09:15:00 | 5952.06 | TARGET | 272.44 |
