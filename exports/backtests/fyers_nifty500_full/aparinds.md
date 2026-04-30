# Apar Industries Ltd. (APARINDS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 12368.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -1094.45
- **Avg P&L per closed trade:** -273.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 7179.00 | 9892.08 | 9894.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 09:15:00 | 7006.30 | 9863.37 | 9880.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 5507.50 | 5482.67 | 6228.20 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 7567.50 | 6392.56 | 6387.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 7647.00 | 6405.04 | 6393.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 8654.00 | 8663.33 | 8166.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 9330.00 | 8668.52 | 8181.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-19 10:15:00 | 8420.00 | 8745.49 | 8426.92 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 7803.00 | 8250.77 | 8251.85 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 8551.00 | 8253.80 | 8252.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 8559.50 | 8261.16 | 8256.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 8486.50 | 8508.93 | 8403.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-16 09:15:00 | 8655.50 | 8441.50 | 8398.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 8655.50 | 8441.50 | 8398.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-16 13:15:00 | 8883.00 | 8456.37 | 8406.84 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-03 10:15:00 | 8544.00 | 8685.89 | 8551.81 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 8350.00 | 8725.95 | 8727.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 8253.00 | 8717.35 | 8722.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 7925.50 | 7851.31 | 8181.49 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 9500.00 | 8407.66 | 8405.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 9575.00 | 8472.66 | 8438.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 9782.00 | 9811.86 | 9322.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-25 09:15:00 | 9950.00 | 9654.12 | 9367.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 9333.50 | 9741.54 | 9451.97 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-29 12:15:00 | 9330.00 | 2025-08-19 10:15:00 | 8420.00 | EXIT_EMA400 | -910.00 |
| BUY | 2025-10-16 09:15:00 | 8655.50 | 2025-10-29 10:15:00 | 9426.55 | TARGET | 771.05 |
| BUY | 2025-10-16 13:15:00 | 8883.00 | 2025-11-03 10:15:00 | 8544.00 | EXIT_EMA400 | -339.00 |
| BUY | 2026-03-25 09:15:00 | 9950.00 | 2026-04-02 09:15:00 | 9333.50 | EXIT_EMA400 | -616.50 |
