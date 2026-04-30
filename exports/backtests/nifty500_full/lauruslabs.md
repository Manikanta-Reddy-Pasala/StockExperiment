# Laurus Labs Ltd. (LAURUSLABS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1100.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 48.87
- **Avg P&L per closed trade:** 12.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 361.30 | 383.48 | 383.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 357.25 | 382.76 | 383.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 382.05 | 375.15 | 378.36 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 384.50 | 379.29 | 379.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 386.10 | 379.36 | 379.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 410.60 | 412.02 | 401.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-17 13:15:00 | 412.95 | 412.02 | 401.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 407.55 | 411.97 | 401.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-18 11:15:00 | 413.50 | 411.98 | 401.68 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 406.60 | 412.12 | 402.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-23 11:15:00 | 399.80 | 412.00 | 402.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 392.15 | 399.34 | 399.35 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 10:15:00 | 410.55 | 399.45 | 399.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 11:15:00 | 415.05 | 399.60 | 399.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 14:15:00 | 424.00 | 424.98 | 415.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-29 12:15:00 | 442.05 | 425.84 | 416.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 430.20 | 437.94 | 429.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-30 14:15:00 | 427.05 | 437.83 | 429.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 436.60 | 439.33 | 439.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 432.65 | 439.16 | 439.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 442.25 | 439.10 | 439.21 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 444.45 | 439.34 | 439.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 448.90 | 439.50 | 439.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 473.60 | 474.77 | 461.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-28 12:15:00 | 493.05 | 465.82 | 462.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-15 09:15:00 | 545.15 | 577.38 | 551.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 944.90 | 1015.16 | 1015.40 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1058.45 | 1014.14 | 1014.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1076.95 | 1017.39 | 1015.78 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-17 13:15:00 | 412.95 | 2024-01-23 11:15:00 | 399.80 | EXIT_EMA400 | -13.15 |
| BUY | 2024-01-18 11:15:00 | 413.50 | 2024-01-23 11:15:00 | 399.80 | EXIT_EMA400 | -13.70 |
| BUY | 2024-04-29 12:15:00 | 442.05 | 2024-05-30 14:15:00 | 427.05 | EXIT_EMA400 | -15.00 |
| BUY | 2024-10-28 12:15:00 | 493.05 | 2024-12-03 13:15:00 | 583.77 | TARGET | 90.72 |
