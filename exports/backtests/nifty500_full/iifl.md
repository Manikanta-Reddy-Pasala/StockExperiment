# IIFL Finance Ltd. (IIFL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 458.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -176.55
- **Avg P&L per closed trade:** -35.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 605.55 | 618.71 | 618.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 12:15:00 | 598.20 | 618.35 | 618.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 13:15:00 | 609.00 | 604.92 | 611.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-28 10:15:00 | 573.70 | 605.04 | 609.75 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 606.20 | 602.71 | 608.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-01 11:15:00 | 610.95 | 602.79 | 608.23 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 474.85 | 436.65 | 436.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 12:15:00 | 485.90 | 444.37 | 440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 475.45 | 481.34 | 465.80 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 430.35 | 458.44 | 458.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 11:15:00 | 427.25 | 456.51 | 457.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 14:15:00 | 452.05 | 443.04 | 449.93 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 13:15:00 | 469.60 | 454.17 | 454.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 471.30 | 455.71 | 454.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 483.25 | 483.96 | 472.25 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 420.95 | 465.95 | 466.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 417.50 | 465.02 | 465.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 13:15:00 | 445.75 | 445.55 | 454.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 09:15:00 | 440.60 | 446.93 | 453.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 447.10 | 430.25 | 440.54 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 375.15 | 349.03 | 348.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 387.45 | 351.51 | 350.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 498.55 | 500.73 | 469.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 14:15:00 | 513.60 | 500.82 | 470.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 477.95 | 500.25 | 472.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 455.50 | 499.81 | 472.32 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 437.90 | 459.84 | 459.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 435.55 | 458.35 | 459.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 451.75 | 450.85 | 454.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 12:15:00 | 446.50 | 450.79 | 454.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-18 11:15:00 | 454.30 | 450.29 | 454.08 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 495.35 | 454.74 | 454.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 463.17 | 459.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 10:15:00 | 611.10 | 613.46 | 582.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-21 12:15:00 | 624.90 | 613.56 | 582.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 593.15 | 613.41 | 583.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-22 14:15:00 | 558.00 | 612.20 | 583.63 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 518.00 | 564.51 | 564.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 501.05 | 554.41 | 559.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 469.20 | 465.32 | 490.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 09:15:00 | 447.75 | 466.78 | 487.01 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-28 10:15:00 | 573.70 | 2024-03-01 11:15:00 | 610.95 | EXIT_EMA400 | -37.25 |
| SELL | 2024-11-11 09:15:00 | 440.60 | 2024-12-04 09:15:00 | 447.10 | EXIT_EMA400 | -6.50 |
| BUY | 2025-07-29 14:15:00 | 513.60 | 2025-08-01 09:15:00 | 455.50 | EXIT_EMA400 | -58.10 |
| SELL | 2025-09-16 12:15:00 | 446.50 | 2025-09-18 11:15:00 | 454.30 | EXIT_EMA400 | -7.80 |
| BUY | 2026-01-21 12:15:00 | 624.90 | 2026-01-22 14:15:00 | 558.00 | EXIT_EMA400 | -66.90 |
