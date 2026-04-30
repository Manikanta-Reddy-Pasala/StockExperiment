# Aarti Industries Ltd. (AARTIIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 507.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -70.55
- **Avg P&L per closed trade:** -8.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 518.50 | 484.06 | 483.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 11:15:00 | 520.50 | 484.42 | 484.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 15:15:00 | 496.20 | 496.24 | 491.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-26 09:15:00 | 500.85 | 496.28 | 491.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 493.20 | 496.24 | 491.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-28 12:15:00 | 490.40 | 496.18 | 491.66 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 478.25 | 489.07 | 489.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 475.25 | 488.93 | 489.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 499.20 | 472.67 | 479.46 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 517.65 | 485.14 | 485.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 522.50 | 485.83 | 485.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 598.20 | 599.59 | 565.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-09 12:15:00 | 606.30 | 599.75 | 566.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 637.60 | 659.06 | 638.37 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 621.05 | 675.99 | 676.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 616.90 | 674.85 | 675.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 12:15:00 | 654.60 | 648.94 | 659.86 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 709.35 | 666.83 | 666.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 718.00 | 675.64 | 671.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 687.75 | 688.33 | 679.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-12 10:15:00 | 707.65 | 689.77 | 680.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 685.10 | 693.28 | 683.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-19 12:15:00 | 680.00 | 693.06 | 683.88 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 622.30 | 688.47 | 688.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 618.70 | 676.06 | 682.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 425.80 | 425.31 | 456.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 15:15:00 | 408.00 | 424.84 | 454.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 453.70 | 427.68 | 451.00 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 440.85 | 414.91 | 414.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 448.25 | 415.24 | 414.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 464.80 | 466.02 | 450.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 09:15:00 | 468.00 | 459.77 | 451.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 456.55 | 467.70 | 458.08 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 433.55 | 452.93 | 452.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 429.80 | 452.70 | 452.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 450.00 | 449.29 | 450.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 14:15:00 | 445.30 | 449.25 | 450.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 388.20 | 381.26 | 389.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 11:15:00 | 389.40 | 381.41 | 389.32 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 13:15:00 | 447.45 | 375.13 | 374.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 454.90 | 375.93 | 375.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 417.50 | 428.97 | 410.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 14:15:00 | 427.90 | 425.11 | 411.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 413.90 | 426.87 | 413.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 11:15:00 | 424.35 | 426.74 | 413.55 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-19 13:15:00 | 414.35 | 426.31 | 414.75 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-26 09:15:00 | 500.85 | 2023-09-28 12:15:00 | 490.40 | EXIT_EMA400 | -10.45 |
| BUY | 2024-01-09 12:15:00 | 606.30 | 2024-03-13 09:15:00 | 637.60 | EXIT_EMA400 | 31.30 |
| BUY | 2024-07-12 10:15:00 | 707.65 | 2024-07-19 12:15:00 | 680.00 | EXIT_EMA400 | -27.65 |
| SELL | 2025-01-13 15:15:00 | 408.00 | 2025-01-21 09:15:00 | 453.70 | EXIT_EMA400 | -45.70 |
| BUY | 2025-06-25 09:15:00 | 468.00 | 2025-07-08 10:15:00 | 456.55 | EXIT_EMA400 | -11.45 |
| SELL | 2025-07-30 14:15:00 | 445.30 | 2025-07-31 09:15:00 | 428.35 | TARGET | 16.95 |
| BUY | 2026-03-10 14:15:00 | 427.90 | 2026-03-19 13:15:00 | 414.35 | EXIT_EMA400 | -13.55 |
| BUY | 2026-03-16 11:15:00 | 424.35 | 2026-03-19 13:15:00 | 414.35 | EXIT_EMA400 | -10.00 |
