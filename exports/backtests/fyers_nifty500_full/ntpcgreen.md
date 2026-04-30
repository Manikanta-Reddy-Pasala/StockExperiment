# NTPC Green Energy Ltd. (NTPCGREEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-27 09:15:00 → 2026-04-30 15:15:00 (2466 bars)
- **Last close:** 110.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 10.32
- **Avg P&L per closed trade:** 3.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 113.01 | 104.28 | 104.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 113.56 | 104.55 | 104.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 107.92 | 108.10 | 106.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 108.24 | 108.06 | 106.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 107.25 | 108.02 | 106.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 12:15:00 | 106.44 | 107.98 | 106.72 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 104.60 | 107.11 | 107.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 104.36 | 107.08 | 107.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 105.00 | 104.86 | 105.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 09:15:00 | 104.53 | 104.86 | 105.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 105.07 | 104.20 | 105.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-02 11:15:00 | 105.24 | 104.21 | 105.22 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 11:15:00 | 100.48 | 91.55 | 91.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 101.43 | 92.03 | 91.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 14:15:00 | 92.84 | 93.42 | 92.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-01 09:15:00 | 96.90 | 93.44 | 92.62 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-16 12:15:00 | 108.24 | 2025-06-18 12:15:00 | 106.44 | EXIT_EMA400 | -1.80 |
| SELL | 2025-08-21 09:15:00 | 104.53 | 2025-09-02 11:15:00 | 105.24 | EXIT_EMA400 | -0.71 |
| BUY | 2026-04-01 09:15:00 | 96.90 | 2026-04-16 10:15:00 | 109.73 | TARGET | 12.83 |
