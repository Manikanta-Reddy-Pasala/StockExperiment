# Gabriel India Ltd. (GABRIEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1025.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** -37.25
- **Avg P&L per closed trade:** -4.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 14:15:00 | 359.00 | 379.07 | 379.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 15:15:00 | 356.95 | 378.85 | 378.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 12:15:00 | 343.85 | 343.20 | 357.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-18 13:15:00 | 331.65 | 343.08 | 356.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-01 14:15:00 | 360.85 | 339.81 | 351.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 14:15:00 | 386.90 | 356.12 | 356.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 388.10 | 365.73 | 361.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 11:15:00 | 368.95 | 369.59 | 364.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 09:15:00 | 373.25 | 367.28 | 364.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 373.25 | 367.28 | 364.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 10:15:00 | 375.70 | 367.36 | 364.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 504.40 | 518.89 | 504.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-03 15:15:00 | 503.00 | 518.59 | 504.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 445.80 | 493.69 | 493.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 442.10 | 491.82 | 492.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 464.40 | 462.34 | 474.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-05 14:15:00 | 460.15 | 462.31 | 474.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 456.10 | 445.26 | 457.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-04 09:15:00 | 448.75 | 445.30 | 457.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-06 09:15:00 | 472.00 | 445.28 | 456.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 514.95 | 465.59 | 465.44 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 435.65 | 470.60 | 470.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 434.00 | 470.23 | 470.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 12:15:00 | 483.30 | 448.18 | 457.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 09:15:00 | 442.45 | 448.26 | 457.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 454.00 | 448.27 | 457.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 13:15:00 | 460.00 | 448.47 | 457.41 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 498.20 | 464.50 | 464.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 506.00 | 465.44 | 464.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-17 09:15:00 | 464.70 | 467.62 | 466.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-21 14:15:00 | 480.85 | 466.26 | 465.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 466.10 | 469.87 | 467.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1028.90 | 1175.11 | 1175.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1025.30 | 1151.34 | 1163.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1052.40 | 1038.33 | 1088.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 11:15:00 | 1017.50 | 1041.46 | 1081.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1074.50 | 1037.08 | 1072.77 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1019.70 | 945.76 | 945.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1024.45 | 950.23 | 947.94 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-18 13:15:00 | 331.65 | 2024-04-01 14:15:00 | 360.85 | EXIT_EMA400 | -29.20 |
| BUY | 2024-06-05 09:15:00 | 373.25 | 2024-06-11 11:15:00 | 400.95 | TARGET | 27.70 |
| BUY | 2024-06-05 10:15:00 | 375.70 | 2024-06-11 11:15:00 | 410.58 | TARGET | 34.88 |
| SELL | 2024-11-05 14:15:00 | 460.15 | 2024-11-13 09:15:00 | 418.23 | TARGET | 41.92 |
| SELL | 2024-12-04 09:15:00 | 448.75 | 2024-12-06 09:15:00 | 472.00 | EXIT_EMA400 | -23.25 |
| SELL | 2025-01-31 09:15:00 | 442.45 | 2025-01-31 13:15:00 | 460.00 | EXIT_EMA400 | -17.55 |
| BUY | 2025-02-21 14:15:00 | 480.85 | 2025-02-28 09:15:00 | 466.10 | EXIT_EMA400 | -14.75 |
| SELL | 2025-12-29 11:15:00 | 1017.50 | 2026-01-05 09:15:00 | 1074.50 | EXIT_EMA400 | -57.00 |
