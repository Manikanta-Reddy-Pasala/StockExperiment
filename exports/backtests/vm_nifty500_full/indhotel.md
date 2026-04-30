# Indian Hotels Co. Ltd. (INDHOTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 635.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -6.17
- **Avg P&L per closed trade:** -0.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 14:15:00 | 393.10 | 390.19 | 390.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 400.10 | 390.60 | 390.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 13:15:00 | 409.00 | 410.65 | 403.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-26 10:15:00 | 412.35 | 410.01 | 403.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 404.60 | 410.22 | 404.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-09-29 09:15:00 | 408.40 | 410.16 | 404.29 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-10-04 12:15:00 | 402.85 | 410.20 | 404.79 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 382.20 | 404.74 | 404.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 380.35 | 404.07 | 404.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 401.80 | 400.36 | 402.37 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 15:15:00 | 411.90 | 403.79 | 403.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 417.95 | 403.93 | 403.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 425.30 | 427.98 | 419.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 435.95 | 427.88 | 419.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 564.10 | 585.43 | 562.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-26 14:15:00 | 567.20 | 585.25 | 562.22 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-07 12:15:00 | 561.30 | 582.14 | 564.73 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 721.80 | 786.61 | 786.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 715.90 | 785.91 | 786.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 14:15:00 | 757.75 | 753.58 | 766.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-06 11:15:00 | 747.10 | 753.56 | 766.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-13 10:15:00 | 763.75 | 752.11 | 763.47 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 09:15:00 | 844.05 | 772.08 | 771.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 761.80 | 785.24 | 785.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 760.30 | 782.13 | 783.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 779.85 | 777.36 | 780.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-27 09:15:00 | 765.25 | 777.12 | 780.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-02 14:15:00 | 784.00 | 774.38 | 778.44 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 807.85 | 760.98 | 760.77 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 709.70 | 765.45 | 765.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 709.00 | 741.98 | 748.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 728.25 | 726.78 | 737.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 13:15:00 | 718.35 | 731.44 | 736.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-12 09:15:00 | 737.70 | 730.05 | 734.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-26 10:15:00 | 412.35 | 2023-10-04 12:15:00 | 402.85 | EXIT_EMA400 | -9.50 |
| BUY | 2023-09-29 09:15:00 | 408.40 | 2023-10-04 12:15:00 | 402.85 | EXIT_EMA400 | -5.55 |
| BUY | 2023-12-22 09:15:00 | 435.95 | 2024-01-23 09:15:00 | 484.65 | TARGET | 48.70 |
| BUY | 2024-04-26 14:15:00 | 567.20 | 2024-04-29 09:15:00 | 582.13 | TARGET | 14.93 |
| SELL | 2025-03-06 11:15:00 | 747.10 | 2025-03-13 10:15:00 | 763.75 | EXIT_EMA400 | -16.65 |
| SELL | 2025-05-27 09:15:00 | 765.25 | 2025-06-02 14:15:00 | 784.00 | EXIT_EMA400 | -18.75 |
| SELL | 2025-12-08 13:15:00 | 718.35 | 2025-12-12 09:15:00 | 737.70 | EXIT_EMA400 | -19.35 |
