# Abbott India Ltd. (ABBOTINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 25435.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 2590.75
- **Avg P&L per closed trade:** 370.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 10:15:00 | 22416.05 | 23114.49 | 23114.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 15:15:00 | 22280.00 | 22995.08 | 23051.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 23000.00 | 22988.95 | 23047.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-04 12:15:00 | 22739.90 | 22998.47 | 23048.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 22974.50 | 22984.65 | 23038.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-06 11:15:00 | 23092.95 | 22985.09 | 23038.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 10:15:00 | 23937.30 | 22917.13 | 22912.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 22364.75 | 23094.62 | 23096.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 22125.05 | 23084.97 | 23091.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 09:15:00 | 23022.10 | 22884.95 | 22974.83 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 10:15:00 | 23865.55 | 23051.06 | 23050.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 24050.00 | 23119.09 | 23085.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 27749.95 | 27791.89 | 26523.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 09:15:00 | 28046.05 | 27613.11 | 26685.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-22 13:15:00 | 26907.05 | 27747.04 | 26955.45 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 13:15:00 | 25592.95 | 26692.34 | 26695.69 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 12:15:00 | 27890.70 | 26507.06 | 26504.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 13:15:00 | 27991.75 | 26521.83 | 26512.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 26846.15 | 26866.88 | 26710.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-21 09:15:00 | 27331.20 | 26871.02 | 26717.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-24 09:15:00 | 26710.00 | 26879.77 | 26727.60 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 27457.85 | 28602.80 | 28603.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 27317.05 | 28577.84 | 28591.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 28150.60 | 28094.40 | 28305.98 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 28648.75 | 28453.44 | 28452.94 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 14:15:00 | 28243.75 | 28450.42 | 28451.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 28152.00 | 28445.49 | 28448.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 28572.50 | 28408.13 | 28429.25 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 29016.95 | 28451.65 | 28450.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 29488.95 | 28490.79 | 28471.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 11:15:00 | 29122.05 | 29140.78 | 28860.65 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 27773.70 | 28653.83 | 28654.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 27659.50 | 28643.94 | 28649.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 13:15:00 | 27530.00 | 27470.35 | 27966.73 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 29006.75 | 28298.56 | 28298.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 13:15:00 | 29448.00 | 28368.29 | 28333.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 29601.50 | 29656.49 | 29127.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-17 11:15:00 | 29829.05 | 29657.88 | 29138.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-02 14:15:00 | 29560.00 | 30121.34 | 29583.98 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 31030.00 | 32680.45 | 32688.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 30945.00 | 32555.52 | 32623.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 29845.00 | 29694.59 | 30210.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 10:15:00 | 29610.00 | 29723.74 | 30197.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 30010.00 | 29707.75 | 30161.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 09:15:00 | 29660.00 | 29707.28 | 30159.33 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 28950.00 | 28673.02 | 29219.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-01 11:15:00 | 28640.00 | 28677.35 | 29210.54 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-04 13:15:00 | 27555.00 | 26932.29 | 27539.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-04 12:15:00 | 22739.90 | 2023-10-06 11:15:00 | 23092.95 | EXIT_EMA400 | -353.05 |
| BUY | 2024-03-14 09:15:00 | 28046.05 | 2024-03-22 13:15:00 | 26907.05 | EXIT_EMA400 | -1139.00 |
| BUY | 2024-06-21 09:15:00 | 27331.20 | 2024-06-24 09:15:00 | 26710.00 | EXIT_EMA400 | -621.20 |
| BUY | 2025-03-17 11:15:00 | 29829.05 | 2025-04-02 14:15:00 | 29560.00 | EXIT_EMA400 | -269.05 |
| SELL | 2025-12-01 09:15:00 | 29660.00 | 2025-12-09 09:15:00 | 28162.01 | TARGET | 1497.99 |
| SELL | 2025-11-27 10:15:00 | 29610.00 | 2025-12-10 10:15:00 | 27846.56 | TARGET | 1763.44 |
| SELL | 2026-01-01 11:15:00 | 28640.00 | 2026-01-29 09:15:00 | 26928.38 | TARGET | 1711.62 |
