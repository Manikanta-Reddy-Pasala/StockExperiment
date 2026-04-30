# Marico Ltd. (MARICO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 776.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -96.85
- **Avg P&L per closed trade:** -16.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 11:15:00 | 654.60 | 667.66 | 667.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 12:15:00 | 647.70 | 667.46 | 667.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 631.50 | 628.89 | 643.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 605.80 | 633.11 | 642.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-12 09:15:00 | 640.90 | 630.30 | 640.10 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 15:15:00 | 673.55 | 642.29 | 642.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 676.50 | 654.55 | 649.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 658.60 | 662.21 | 655.06 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 626.45 | 649.87 | 649.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 623.10 | 647.47 | 648.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 620.75 | 620.53 | 631.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-19 10:15:00 | 618.50 | 620.52 | 631.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 630.55 | 620.64 | 630.69 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 12:15:00 | 630.85 | 620.74 | 630.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 13:15:00 | 662.65 | 636.42 | 636.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 669.30 | 637.26 | 636.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 704.45 | 706.03 | 685.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-21 09:15:00 | 716.60 | 706.11 | 685.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 10:15:00 | 693.35 | 705.81 | 694.60 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 693.70 | 716.49 | 716.58 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 731.10 | 716.12 | 716.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 735.10 | 716.99 | 716.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 10:15:00 | 723.00 | 718.57 | 717.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 719.70 | 718.72 | 717.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-30 15:15:00 | 723.80 | 718.88 | 717.69 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 719.75 | 719.15 | 717.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-04 10:15:00 | 717.75 | 719.26 | 717.98 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 753.15 | 755.64 | 755.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 11:15:00 | 748.00 | 755.52 | 755.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-16 12:15:00 | 741.95 | 755.10 | 755.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-17 10:15:00 | 756.80 | 754.78 | 755.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 762.80 | 755.53 | 755.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 767.60 | 755.73 | 755.63 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 09:15:00 | 605.80 | 2024-12-12 09:15:00 | 640.90 | EXIT_EMA400 | -35.10 |
| SELL | 2025-03-19 10:15:00 | 618.50 | 2025-03-20 12:15:00 | 630.85 | EXIT_EMA400 | -12.35 |
| BUY | 2025-05-21 09:15:00 | 716.60 | 2025-06-12 10:15:00 | 693.35 | EXIT_EMA400 | -23.25 |
| BUY | 2025-10-29 10:15:00 | 723.00 | 2025-11-04 10:15:00 | 717.75 | EXIT_EMA400 | -5.25 |
| BUY | 2025-10-30 15:15:00 | 723.80 | 2025-11-04 10:15:00 | 717.75 | EXIT_EMA400 | -6.05 |
| SELL | 2026-04-16 12:15:00 | 741.95 | 2026-04-17 10:15:00 | 756.80 | EXIT_EMA400 | -14.85 |
