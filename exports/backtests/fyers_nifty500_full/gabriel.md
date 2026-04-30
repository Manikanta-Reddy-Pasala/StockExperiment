# Gabriel India Ltd. (GABRIEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1024.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -69.43
- **Avg P&L per closed trade:** -13.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 445.80 | 493.76 | 493.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 442.20 | 491.91 | 492.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 464.80 | 462.32 | 474.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-05 15:15:00 | 459.50 | 462.26 | 473.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 456.10 | 445.21 | 457.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-04 09:15:00 | 448.25 | 445.24 | 457.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-06 09:15:00 | 472.10 | 445.25 | 456.42 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 514.95 | 465.54 | 465.36 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 435.65 | 470.61 | 470.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 434.00 | 470.24 | 470.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 12:15:00 | 483.65 | 448.19 | 457.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 09:15:00 | 442.45 | 448.27 | 457.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 454.00 | 448.28 | 457.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 13:15:00 | 460.00 | 448.47 | 457.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 10:15:00 | 472.90 | 463.76 | 463.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 11:15:00 | 475.60 | 463.88 | 463.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-17 09:15:00 | 464.25 | 467.67 | 465.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-21 14:15:00 | 480.35 | 466.29 | 465.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 466.10 | 469.90 | 467.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1028.90 | 1175.02 | 1175.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1025.30 | 1151.26 | 1163.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1052.40 | 1038.28 | 1088.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 11:15:00 | 1017.50 | 1041.45 | 1081.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1074.60 | 1037.09 | 1072.75 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1013.90 | 944.99 | 944.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1020.35 | 947.72 | 946.29 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-05 15:15:00 | 459.50 | 2024-11-13 09:15:00 | 416.18 | TARGET | 43.32 |
| SELL | 2024-12-04 09:15:00 | 448.25 | 2024-12-06 09:15:00 | 472.10 | EXIT_EMA400 | -23.85 |
| SELL | 2025-01-31 09:15:00 | 442.45 | 2025-01-31 13:15:00 | 460.00 | EXIT_EMA400 | -17.55 |
| BUY | 2025-02-21 14:15:00 | 480.35 | 2025-02-28 09:15:00 | 466.10 | EXIT_EMA400 | -14.25 |
| SELL | 2025-12-29 11:15:00 | 1017.50 | 2026-01-05 09:15:00 | 1074.60 | EXIT_EMA400 | -57.10 |
