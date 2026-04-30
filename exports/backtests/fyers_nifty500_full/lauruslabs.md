# Laurus Labs Ltd. (LAURUSLABS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1101.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 28.88
- **Avg P&L per closed trade:** 9.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 424.80 | 440.60 | 440.62 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 14:15:00 | 446.25 | 440.50 | 440.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 456.25 | 441.13 | 440.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 473.80 | 474.75 | 461.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-28 12:15:00 | 493.05 | 465.83 | 462.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-15 09:15:00 | 545.15 | 577.47 | 552.06 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 546.80 | 561.36 | 561.43 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 586.30 | 561.73 | 561.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 589.75 | 566.96 | 564.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 571.05 | 590.84 | 579.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 14:15:00 | 625.30 | 588.10 | 580.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-08 12:15:00 | 597.30 | 610.99 | 598.21 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 976.95 | 1006.22 | 1006.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 966.30 | 1005.83 | 1006.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.41 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1017.00 | 1006.20 | 1006.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1024.10 | 1007.29 | 1006.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 11:15:00 | 1026.40 | 1027.67 | 1018.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-04 12:15:00 | 1030.60 | 1027.70 | 1018.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 997.10 | 1028.98 | 1019.81 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 971.40 | 1014.23 | 1014.44 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1059.00 | 1013.43 | 1013.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1061.05 | 1014.34 | 1013.82 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-28 12:15:00 | 493.05 | 2024-12-03 12:15:00 | 583.43 | TARGET | 90.38 |
| BUY | 2025-04-15 14:15:00 | 625.30 | 2025-05-08 12:15:00 | 597.30 | EXIT_EMA400 | -28.00 |
| BUY | 2026-03-04 12:15:00 | 1030.60 | 2026-03-09 09:15:00 | 997.10 | EXIT_EMA400 | -33.50 |
