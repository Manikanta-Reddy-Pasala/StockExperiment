# Amber Enterprises India Ltd. (AMBER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 8009.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -80.28
- **Avg P&L per closed trade:** -13.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 11:15:00 | 5487.20 | 6480.22 | 6482.69 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 09:15:00 | 6944.30 | 6371.01 | 6370.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 7297.85 | 6540.23 | 6462.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 6625.00 | 6648.29 | 6532.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 6856.00 | 6570.17 | 6506.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 6590.00 | 6622.79 | 6548.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-24 09:15:00 | 6453.50 | 6621.10 | 6547.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 14:15:00 | 6104.00 | 6492.29 | 6492.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 6012.00 | 6483.57 | 6488.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 6350.00 | 6344.35 | 6406.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-19 09:15:00 | 6236.50 | 6351.05 | 6406.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 6236.50 | 6351.05 | 6406.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-20 09:15:00 | 6502.50 | 6348.13 | 6403.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 6575.00 | 6433.30 | 6432.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 6626.50 | 6446.86 | 6440.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 6561.50 | 6460.92 | 6448.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 6561.50 | 6460.92 | 6448.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-23 11:15:00 | 6710.50 | 6464.64 | 6450.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-11 09:15:00 | 7053.00 | 7489.00 | 7203.44 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 7327.50 | 7824.65 | 7825.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7708.54 | 7763.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 15:15:00 | 6165.00 | 6144.57 | 6512.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 6580.50 | 6184.87 | 6503.42 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 7815.00 | 6741.32 | 6739.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 7909.50 | 6804.35 | 6771.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 7312.50 | 7367.95 | 7122.65 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 6567.50 | 6990.22 | 6991.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6482.00 | 6962.72 | 6977.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 6782.50 | 6817.79 | 6891.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-09 14:15:00 | 6900.00 | 6818.61 | 6891.96 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 7715.00 | 6955.85 | 6954.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 7874.00 | 6964.99 | 6959.11 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-15 09:15:00 | 6856.00 | 2025-04-24 09:15:00 | 6453.50 | EXIT_EMA400 | -402.50 |
| SELL | 2025-05-19 09:15:00 | 6236.50 | 2025-05-20 09:15:00 | 6502.50 | EXIT_EMA400 | -266.00 |
| BUY | 2025-06-23 09:15:00 | 6561.50 | 2025-06-25 11:15:00 | 6901.60 | TARGET | 340.10 |
| BUY | 2025-06-23 11:15:00 | 6710.50 | 2025-07-08 09:15:00 | 7491.62 | TARGET | 781.12 |
| SELL | 2026-02-03 15:15:00 | 6165.00 | 2026-02-06 11:15:00 | 6580.50 | EXIT_EMA400 | -415.50 |
| SELL | 2026-04-09 13:15:00 | 6782.50 | 2026-04-09 14:15:00 | 6900.00 | EXIT_EMA400 | -117.50 |
