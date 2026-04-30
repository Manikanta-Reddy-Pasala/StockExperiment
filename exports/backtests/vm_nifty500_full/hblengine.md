# HBL Engineering Ltd. (HBLENGINE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-23 09:15:00 → 2026-04-30 15:15:00 (2322 bars)
- **Last close:** 798.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 80.95
- **Avg P&L per closed trade:** 26.98

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 569.70 | 507.03 | 506.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 578.10 | 508.35 | 507.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 577.35 | 577.45 | 554.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 588.30 | 576.59 | 556.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-23 13:15:00 | 583.75 | 601.53 | 584.59 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 814.25 | 873.99 | 874.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 804.80 | 872.17 | 873.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 862.15 | 855.37 | 863.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 15:15:00 | 845.90 | 855.23 | 863.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 864.30 | 855.32 | 863.64 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 925.55 | 870.33 | 870.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 932.50 | 876.64 | 873.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 896.65 | 896.81 | 885.18 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 757.50 | 875.93 | 876.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 752.95 | 874.71 | 875.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 804.60 | 801.30 | 827.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 11:15:00 | 792.40 | 801.21 | 827.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 14:15:00 | 719.45 | 681.51 | 719.09 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 802.75 | 741.15 | 741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 825.60 | 745.43 | 743.30 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-24 09:15:00 | 588.30 | 2025-07-23 13:15:00 | 583.75 | EXIT_EMA400 | -4.55 |
| SELL | 2025-12-22 15:15:00 | 845.90 | 2025-12-23 09:15:00 | 864.30 | EXIT_EMA400 | -18.40 |
| SELL | 2026-02-11 11:15:00 | 792.40 | 2026-02-27 09:15:00 | 688.50 | TARGET | 103.90 |
