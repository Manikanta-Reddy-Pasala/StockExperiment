# Ramkrishna Forgings Ltd. (RKFORGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 598.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 125.75
- **Avg P&L per closed trade:** 15.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 14:15:00 | 623.80 | 736.88 | 736.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 09:15:00 | 615.15 | 734.55 | 735.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 712.00 | 697.45 | 712.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-05 11:15:00 | 696.55 | 700.64 | 712.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 701.00 | 700.32 | 712.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-09 12:15:00 | 692.30 | 700.24 | 711.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 705.95 | 699.33 | 709.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-15 12:15:00 | 715.95 | 699.57 | 709.94 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 11:15:00 | 773.65 | 718.10 | 717.86 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 13:15:00 | 689.75 | 720.22 | 720.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 677.00 | 714.85 | 717.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 11:15:00 | 718.95 | 707.62 | 713.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 09:15:00 | 693.85 | 707.59 | 713.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 706.55 | 700.49 | 708.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 710.15 | 700.45 | 708.64 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 807.30 | 714.49 | 714.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 821.60 | 716.56 | 715.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 857.20 | 869.23 | 818.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-22 13:15:00 | 885.55 | 869.32 | 821.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 09:15:00 | 943.20 | 981.43 | 946.11 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 904.20 | 952.59 | 952.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 894.05 | 950.58 | 951.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 11:15:00 | 981.55 | 926.36 | 936.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 903.95 | 939.98 | 941.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 750.45 | 715.07 | 765.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-20 15:15:00 | 745.20 | 717.09 | 764.85 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 774.20 | 717.66 | 764.90 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 567.40 | 527.75 | 527.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 571.00 | 528.96 | 528.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 544.00 | 544.24 | 537.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-27 09:15:00 | 551.35 | 544.20 | 538.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 550.05 | 544.64 | 538.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-02 14:15:00 | 557.40 | 545.06 | 538.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 544.50 | 549.05 | 542.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 10:15:00 | 539.50 | 548.96 | 542.41 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 474.35 | 538.19 | 538.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 467.70 | 526.72 | 532.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 525.10 | 519.62 | 527.51 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 551.45 | 532.37 | 532.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 560.90 | 533.38 | 532.79 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-05 11:15:00 | 696.55 | 2024-04-15 12:15:00 | 715.95 | EXIT_EMA400 | -19.40 |
| SELL | 2024-04-09 12:15:00 | 692.30 | 2024-04-15 12:15:00 | 715.95 | EXIT_EMA400 | -23.65 |
| SELL | 2024-06-04 09:15:00 | 693.85 | 2024-06-04 11:15:00 | 636.37 | TARGET | 57.48 |
| BUY | 2024-07-22 13:15:00 | 885.55 | 2024-10-07 09:15:00 | 943.20 | EXIT_EMA400 | 57.65 |
| SELL | 2025-01-22 09:15:00 | 903.95 | 2025-01-27 09:15:00 | 791.53 | TARGET | 112.42 |
| SELL | 2025-03-20 15:15:00 | 745.20 | 2025-03-21 09:15:00 | 774.20 | EXIT_EMA400 | -29.00 |
| BUY | 2026-02-27 09:15:00 | 551.35 | 2026-03-13 10:15:00 | 539.50 | EXIT_EMA400 | -11.85 |
| BUY | 2026-03-02 14:15:00 | 557.40 | 2026-03-13 10:15:00 | 539.50 | EXIT_EMA400 | -17.90 |
