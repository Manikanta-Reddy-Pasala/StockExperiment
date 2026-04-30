# Linde India Ltd. (LINDEINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 7320.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 1
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -1159.23
- **Avg P&L per closed trade:** -115.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 10:15:00 | 5635.05 | 5873.99 | 5874.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 5579.10 | 5852.28 | 5863.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 10:15:00 | 5638.00 | 5634.17 | 5716.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-29 13:15:00 | 5435.00 | 5606.39 | 5652.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 5600.00 | 5578.51 | 5632.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-06 12:15:00 | 5638.60 | 5579.29 | 5632.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-13 13:15:00 | 6194.90 | 5677.05 | 5675.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 10:15:00 | 6376.00 | 5695.13 | 5684.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 8051.60 | 8271.96 | 7598.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-29 11:15:00 | 8495.30 | 8274.93 | 7606.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 8408.50 | 8391.73 | 7759.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-06 09:15:00 | 8455.65 | 8366.82 | 7780.87 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 8238.00 | 8659.53 | 8151.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-24 13:15:00 | 8359.35 | 8656.54 | 8152.50 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 8197.15 | 8624.87 | 8160.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-27 09:15:00 | 8155.05 | 8594.55 | 8161.41 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 7711.90 | 8218.12 | 8220.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 7651.35 | 8212.48 | 8217.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 7579.95 | 7526.29 | 7752.37 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 8680.25 | 7885.36 | 7884.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 10:15:00 | 8732.05 | 7931.15 | 7907.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 8067.00 | 8081.07 | 7994.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-07 09:15:00 | 8189.95 | 8080.66 | 7995.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 10:15:00 | 7900.05 | 8078.86 | 7995.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 7676.00 | 7999.84 | 8000.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 7412.00 | 7911.72 | 7951.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 14:15:00 | 6602.55 | 6537.24 | 6903.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 11:15:00 | 6263.90 | 6520.31 | 6840.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6265.00 | 6138.17 | 6504.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 09:15:00 | 6020.85 | 6180.88 | 6442.22 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-20 09:15:00 | 6356.70 | 5986.60 | 6269.47 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 6356.00 | 6222.27 | 6221.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 6401.50 | 6225.43 | 6223.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 6199.00 | 6272.68 | 6249.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 6356.00 | 6241.49 | 6236.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6356.00 | 6241.49 | 6236.04 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 10:15:00 | 6381.00 | 6242.87 | 6236.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 11:15:00 | 6839.00 | 7129.27 | 6883.15 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 6639.00 | 6773.70 | 6774.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 6606.00 | 6768.95 | 6771.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 6562.00 | 6467.58 | 6574.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 12:15:00 | 6350.50 | 6463.31 | 6567.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-09 09:15:00 | 6535.00 | 6442.47 | 6531.19 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 6601.50 | 6003.54 | 6003.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 13:15:00 | 6909.00 | 6105.19 | 6056.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 10:15:00 | 6597.50 | 6769.60 | 6533.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-23 13:15:00 | 6823.50 | 6767.85 | 6535.93 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-29 13:15:00 | 5435.00 | 2024-03-06 12:15:00 | 5638.60 | EXIT_EMA400 | -203.60 |
| BUY | 2024-05-29 11:15:00 | 8495.30 | 2024-06-27 09:15:00 | 8155.05 | EXIT_EMA400 | -340.25 |
| BUY | 2024-06-06 09:15:00 | 8455.65 | 2024-06-27 09:15:00 | 8155.05 | EXIT_EMA400 | -300.60 |
| BUY | 2024-06-24 13:15:00 | 8359.35 | 2024-06-27 09:15:00 | 8155.05 | EXIT_EMA400 | -204.30 |
| BUY | 2024-10-07 09:15:00 | 8189.95 | 2024-10-07 10:15:00 | 7900.05 | EXIT_EMA400 | -289.90 |
| SELL | 2025-01-13 11:15:00 | 6263.90 | 2025-02-20 09:15:00 | 6356.70 | EXIT_EMA400 | -92.80 |
| SELL | 2025-02-10 09:15:00 | 6020.85 | 2025-02-20 09:15:00 | 6356.70 | EXIT_EMA400 | -335.85 |
| BUY | 2025-05-12 09:15:00 | 6356.00 | 2025-05-14 10:15:00 | 6715.87 | TARGET | 359.87 |
| BUY | 2025-05-12 10:15:00 | 6381.00 | 2025-05-14 12:15:00 | 6813.70 | TARGET | 432.70 |
| SELL | 2025-08-28 12:15:00 | 6350.50 | 2025-09-09 09:15:00 | 6535.00 | EXIT_EMA400 | -184.50 |
