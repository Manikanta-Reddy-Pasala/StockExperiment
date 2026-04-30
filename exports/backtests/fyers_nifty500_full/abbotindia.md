# Abbott India Ltd. (ABBOTINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 25480.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| EXIT | 2 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 4 / 1
- **Total realized P&L (per unit):** 6041.16
- **Avg P&L per closed trade:** 1208.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 27367.85 | 28597.67 | 28598.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 27311.50 | 28584.87 | 28592.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 28157.15 | 28100.19 | 28308.02 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 28567.85 | 28452.51 | 28452.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 28683.35 | 28454.81 | 28453.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 28446.50 | 28458.08 | 28455.27 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 15:15:00 | 28287.00 | 28451.74 | 28452.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 28152.00 | 28448.75 | 28450.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 28577.70 | 28411.01 | 28430.85 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 29016.95 | 28454.50 | 28452.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 29489.05 | 28494.48 | 28473.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 11:15:00 | 29122.05 | 29140.77 | 28861.08 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 27773.70 | 28655.44 | 28655.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 27659.50 | 28645.53 | 28650.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 13:15:00 | 27530.00 | 27377.24 | 27902.40 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 29006.75 | 28255.40 | 28254.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 14:15:00 | 29822.30 | 28344.48 | 28299.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 29600.55 | 29636.08 | 29097.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-17 11:15:00 | 29829.05 | 29638.00 | 29109.34 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-02 14:15:00 | 29560.00 | 30112.77 | 29564.77 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 31030.00 | 32679.58 | 32686.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 30945.00 | 32554.72 | 32622.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 29780.00 | 29700.23 | 30225.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-25 09:15:00 | 29565.00 | 29698.89 | 30222.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 30055.00 | 29702.87 | 30209.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-27 09:15:00 | 29825.00 | 29725.55 | 30200.81 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 30010.00 | 29708.35 | 30161.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 09:15:00 | 29660.00 | 29707.86 | 30159.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 28950.00 | 28673.59 | 29219.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-01 11:15:00 | 28640.00 | 28677.90 | 29210.63 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-04 13:15:00 | 27555.00 | 26924.53 | 27521.45 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-03-17 11:15:00 | 29829.05 | 2025-04-02 14:15:00 | 29560.00 | EXIT_EMA400 | -269.05 |
| SELL | 2025-11-27 09:15:00 | 29825.00 | 2025-12-05 09:15:00 | 28697.56 | TARGET | 1127.44 |
| SELL | 2025-12-01 09:15:00 | 29660.00 | 2025-12-09 09:15:00 | 28161.87 | TARGET | 1498.13 |
| SELL | 2025-11-25 09:15:00 | 29565.00 | 2025-12-17 15:15:00 | 27592.27 | TARGET | 1972.73 |
| SELL | 2026-01-01 11:15:00 | 28640.00 | 2026-01-29 09:15:00 | 26928.10 | TARGET | 1711.90 |
