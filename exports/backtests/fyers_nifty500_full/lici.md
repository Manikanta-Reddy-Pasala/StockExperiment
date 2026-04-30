# Life Insurance Corporation of India (LICI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 799.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -64.50
- **Avg P&L per closed trade:** -21.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 993.65 | 1052.70 | 1052.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 985.95 | 1036.59 | 1043.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 953.50 | 946.32 | 975.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 13:15:00 | 925.90 | 945.88 | 974.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-29 09:15:00 | 969.55 | 926.58 | 954.34 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 857.20 | 804.77 | 804.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 861.20 | 806.36 | 805.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 934.45 | 935.83 | 905.06 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 887.95 | 902.82 | 902.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 877.15 | 902.57 | 902.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 888.70 | 885.47 | 891.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 13:15:00 | 883.10 | 885.44 | 891.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 888.95 | 885.41 | 891.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-19 09:15:00 | 894.20 | 885.75 | 891.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 898.60 | 894.45 | 894.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 901.00 | 894.58 | 894.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 894.00 | 895.31 | 894.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-23 09:15:00 | 903.85 | 894.65 | 894.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 903.85 | 894.65 | 894.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-24 10:15:00 | 894.10 | 895.02 | 894.78 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 868.00 | 898.08 | 898.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 863.90 | 894.03 | 896.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 830.45 | 830.00 | 848.08 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 882.75 | 857.78 | 857.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 885.00 | 858.85 | 858.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 857.50 | 860.92 | 859.43 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 826.35 | 857.91 | 858.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 823.70 | 856.12 | 857.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 791.40 | 785.92 | 811.75 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 13:15:00 | 925.90 | 2024-11-29 09:15:00 | 969.55 | EXIT_EMA400 | -43.65 |
| SELL | 2025-09-16 13:15:00 | 883.10 | 2025-09-19 09:15:00 | 894.20 | EXIT_EMA400 | -11.10 |
| BUY | 2025-10-23 09:15:00 | 903.85 | 2025-10-24 10:15:00 | 894.10 | EXIT_EMA400 | -9.75 |
