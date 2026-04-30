# Bosch Ltd. (BOSCHLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 35995.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -366.14
- **Avg P&L per closed trade:** -45.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 19225.85 | 18716.38 | 18714.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 19409.05 | 18723.27 | 18718.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 10:15:00 | 19010.45 | 19037.58 | 18905.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-26 11:15:00 | 19140.10 | 19042.31 | 18913.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 19141.75 | 19057.87 | 18932.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-03 09:15:00 | 18696.05 | 19058.50 | 18937.94 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 15:15:00 | 32451.85 | 32561.21 | 32561.42 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 32615.95 | 32561.76 | 32561.69 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 32546.85 | 32561.61 | 32561.62 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 32781.90 | 32563.80 | 32562.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 12:15:00 | 33779.90 | 32575.90 | 32568.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 10:15:00 | 32529.00 | 32615.60 | 32589.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-09 14:15:00 | 33129.00 | 32615.11 | 32590.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 35647.60 | 36490.20 | 35416.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-25 14:15:00 | 35974.30 | 36460.60 | 35422.92 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 35460.55 | 36421.47 | 35448.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-29 14:15:00 | 36426.05 | 36404.91 | 35464.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-31 09:15:00 | 35405.30 | 36386.73 | 35496.42 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 34783.75 | 35082.19 | 35083.07 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 35617.85 | 35088.11 | 35085.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 10:15:00 | 35791.15 | 35095.11 | 35089.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 12:15:00 | 35571.75 | 35601.05 | 35391.98 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 33666.45 | 35230.29 | 35230.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 33598.30 | 34878.23 | 35037.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 27565.10 | 27526.56 | 29077.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 14:15:00 | 27402.80 | 27542.14 | 28965.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 28100.00 | 27443.91 | 28244.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 14:15:00 | 28450.00 | 27479.18 | 28242.48 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 30765.00 | 28684.92 | 28682.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 30980.00 | 28707.75 | 28693.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 31435.00 | 31473.46 | 30748.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 09:15:00 | 31750.00 | 31476.22 | 30753.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 38425.00 | 39656.93 | 38475.96 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 36670.00 | 38259.28 | 38259.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 38001.45 | 38124.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 37035.00 | 37022.95 | 37465.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 10:15:00 | 36565.00 | 37000.64 | 37421.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 37215.00 | 36344.87 | 36817.16 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 37700.00 | 37206.03 | 37205.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 38210.00 | 37243.33 | 37224.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 37220.00 | 37266.27 | 37236.92 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 36510.00 | 37205.80 | 37207.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 35935.00 | 37193.15 | 37201.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 36570.00 | 36539.65 | 36824.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 36050.00 | 36534.68 | 36818.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 37290.00 | 36531.40 | 36807.41 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 38290.00 | 34582.34 | 34565.47 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-26 11:15:00 | 19140.10 | 2023-10-03 09:15:00 | 18696.05 | EXIT_EMA400 | -444.05 |
| BUY | 2024-09-09 14:15:00 | 33129.00 | 2024-09-13 09:15:00 | 34745.48 | TARGET | 1616.48 |
| BUY | 2024-10-25 14:15:00 | 35974.30 | 2024-10-31 09:15:00 | 35405.30 | EXIT_EMA400 | -569.00 |
| BUY | 2024-10-29 14:15:00 | 36426.05 | 2024-10-31 09:15:00 | 35405.30 | EXIT_EMA400 | -1020.75 |
| SELL | 2025-03-21 14:15:00 | 27402.80 | 2025-04-23 14:15:00 | 28450.00 | EXIT_EMA400 | -1047.20 |
| BUY | 2025-06-25 09:15:00 | 31750.00 | 2025-07-04 09:15:00 | 34738.39 | TARGET | 2988.39 |
| SELL | 2025-12-08 10:15:00 | 36565.00 | 2026-01-02 09:15:00 | 37215.00 | EXIT_EMA400 | -650.00 |
| SELL | 2026-02-02 09:15:00 | 36050.00 | 2026-02-03 09:15:00 | 37290.00 | EXIT_EMA400 | -1240.00 |
