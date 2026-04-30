# Maruti Suzuki India Ltd. (MARUTI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 13314.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 8 |
| ENTRY1 | 9 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 1
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -668.24
- **Avg P&L per closed trade:** -66.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 9620.05 | 9551.45 | 9551.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 9693.00 | 9554.17 | 9552.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 10155.90 | 10271.95 | 10035.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-05 15:15:00 | 10254.00 | 10259.23 | 10043.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 10270.80 | 10469.25 | 10262.42 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-30 13:15:00 | 10420.00 | 10466.82 | 10264.27 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 10303.10 | 10460.03 | 10270.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-11-01 10:15:00 | 10270.05 | 10458.14 | 10270.68 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 10207.20 | 10356.57 | 10356.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 13:15:00 | 10160.50 | 10352.88 | 10354.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 10219.80 | 10192.54 | 10261.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-17 09:15:00 | 10039.20 | 10191.09 | 10258.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 10117.25 | 10091.79 | 10183.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-31 14:15:00 | 10199.40 | 10092.87 | 10183.51 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 14:15:00 | 10923.80 | 10263.78 | 10261.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 11028.95 | 10414.07 | 10342.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 13:15:00 | 12444.15 | 12454.98 | 11988.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-06 09:15:00 | 12563.70 | 12456.76 | 11996.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 12364.75 | 12604.91 | 12324.01 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 12104.40 | 12599.93 | 12322.92 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 12149.80 | 12431.14 | 12431.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 12123.45 | 12368.28 | 12392.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 12334.70 | 12330.06 | 12368.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-17 09:15:00 | 12200.60 | 12326.44 | 12363.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 12324.00 | 12314.08 | 12354.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-19 13:15:00 | 12371.05 | 12315.54 | 12354.41 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 12670.30 | 12389.67 | 12389.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 13162.35 | 12415.62 | 12402.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 12595.65 | 12638.83 | 12529.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 13:15:00 | 12848.45 | 12627.87 | 12534.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 12569.05 | 12659.46 | 12559.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-14 14:15:00 | 12548.05 | 12656.12 | 12559.38 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 11983.00 | 12481.32 | 12483.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 11790.25 | 12441.11 | 12462.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11331.15 | 11321.49 | 11641.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-10 11:15:00 | 11235.30 | 11317.24 | 11619.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-02 09:15:00 | 11430.10 | 11101.36 | 11361.41 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 11993.00 | 11526.56 | 11525.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 12108.55 | 11532.35 | 11527.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 12399.50 | 12411.45 | 12107.48 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 11:15:00 | 11647.20 | 11987.65 | 11988.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 13:15:00 | 11590.00 | 11960.42 | 11974.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 11944.85 | 11925.92 | 11954.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 09:15:00 | 11746.80 | 11923.11 | 11951.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 11799.00 | 11720.28 | 11823.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 10:15:00 | 11873.00 | 11721.80 | 11824.11 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 12458.00 | 11866.54 | 11865.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 12502.00 | 11872.86 | 11868.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 12322.00 | 12347.43 | 12168.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 10:15:00 | 12469.00 | 12305.39 | 12190.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 12302.00 | 12358.73 | 12237.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 12320.00 | 12358.34 | 12238.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-30 13:15:00 | 12360.00 | 12531.06 | 12380.48 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.10 | 16031.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.09 | 16008.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13627.00 | 13309.02 | 14074.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 13180.00 | 13370.51 | 14030.42 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-05 15:15:00 | 10254.00 | 2023-11-01 10:15:00 | 10270.05 | EXIT_EMA400 | 16.05 |
| BUY | 2023-10-30 13:15:00 | 10420.00 | 2023-11-01 10:15:00 | 10270.05 | EXIT_EMA400 | -149.95 |
| SELL | 2024-01-17 09:15:00 | 10039.20 | 2024-01-31 14:15:00 | 10199.40 | EXIT_EMA400 | -160.20 |
| BUY | 2024-05-06 09:15:00 | 12563.70 | 2024-06-04 10:15:00 | 12104.40 | EXIT_EMA400 | -459.30 |
| SELL | 2024-09-17 09:15:00 | 12200.60 | 2024-09-19 13:15:00 | 12371.05 | EXIT_EMA400 | -170.45 |
| BUY | 2024-10-09 13:15:00 | 12848.45 | 2024-10-14 14:15:00 | 12548.05 | EXIT_EMA400 | -300.40 |
| SELL | 2024-12-10 11:15:00 | 11235.30 | 2025-01-02 09:15:00 | 11430.10 | EXIT_EMA400 | -194.80 |
| SELL | 2025-03-26 09:15:00 | 11746.80 | 2025-04-07 09:15:00 | 11132.49 | TARGET | 614.31 |
| BUY | 2025-06-13 10:15:00 | 12320.00 | 2025-06-16 11:15:00 | 12565.51 | TARGET | 245.51 |
| BUY | 2025-06-06 10:15:00 | 12469.00 | 2025-06-30 13:15:00 | 12360.00 | EXIT_EMA400 | -109.00 |
