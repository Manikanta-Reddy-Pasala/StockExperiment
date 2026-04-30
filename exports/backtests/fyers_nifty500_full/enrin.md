# Siemens Energy India Ltd. (ENRIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-06-19 09:15:00 → 2026-04-30 15:15:00 (1493 bars)
- **Last close:** 3280.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -438.10
- **Avg P&L per closed trade:** -146.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 3131.00 | 3261.05 | 3261.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 3126.00 | 3259.71 | 3260.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 11:15:00 | 3236.20 | 3232.79 | 3245.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 3170.00 | 3232.65 | 3245.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3170.00 | 3232.65 | 3245.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-06 10:15:00 | 3160.00 | 3231.93 | 3244.69 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3226.00 | 3227.84 | 3241.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-10 09:15:00 | 3268.80 | 3228.26 | 3241.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 3350.00 | 3252.87 | 3252.51 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 10:15:00 | 3121.00 | 3253.60 | 3253.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 11:15:00 | 3110.70 | 3252.18 | 3253.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 2508.00 | 2493.38 | 2706.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-30 10:15:00 | 2445.10 | 2492.64 | 2703.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 09:15:00 | 2675.60 | 2504.08 | 2670.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 2955.00 | 2742.06 | 2741.31 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 2599.00 | 2759.36 | 2760.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 2540.60 | 2747.73 | 2754.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 2770.30 | 2722.34 | 2739.36 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 11:15:00 | 2892.00 | 2753.92 | 2753.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 2963.50 | 2761.77 | 2757.73 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-11-06 09:15:00 | 3170.00 | 2025-11-10 09:15:00 | 3268.80 | EXIT_EMA400 | -98.80 |
| SELL | 2025-11-06 10:15:00 | 3160.00 | 2025-11-10 09:15:00 | 3268.80 | EXIT_EMA400 | -108.80 |
| SELL | 2026-01-30 10:15:00 | 2445.10 | 2026-02-06 09:15:00 | 2675.60 | EXIT_EMA400 | -230.50 |
