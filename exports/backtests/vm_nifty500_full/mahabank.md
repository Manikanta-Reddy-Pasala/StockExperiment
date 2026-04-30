# Bank of Maharashtra (MAHABANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 78.37
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 2.62
- **Avg P&L per closed trade:** 0.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 61.61 | 65.04 | 65.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 61.22 | 65.00 | 65.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 11:15:00 | 61.24 | 61.09 | 62.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 09:15:00 | 60.30 | 61.17 | 62.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 09:15:00 | 56.12 | 53.79 | 55.74 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 51.19 | 48.89 | 48.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 52.28 | 49.64 | 49.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 53.45 | 53.47 | 52.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 53.86 | 53.47 | 52.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 54.20 | 56.17 | 54.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 52.49 | 54.57 | 54.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 52.33 | 54.38 | 54.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 54.04 | 53.98 | 54.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 11:15:00 | 53.56 | 54.02 | 54.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 54.40 | 54.02 | 54.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 57.10 | 54.45 | 54.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 57.26 | 54.48 | 54.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 09:15:00 | 56.06 | 55.14 | 54.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 56.06 | 55.14 | 54.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 11:15:00 | 56.79 | 55.28 | 54.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 14:15:00 | 54.97 | 55.98 | 55.42 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-25 09:15:00 | 60.30 | 2024-10-10 14:15:00 | 54.27 | TARGET | 6.03 |
| BUY | 2025-06-20 09:15:00 | 53.86 | 2025-08-01 14:15:00 | 54.20 | EXIT_EMA400 | 0.34 |
| SELL | 2025-09-12 11:15:00 | 53.56 | 2025-09-15 09:15:00 | 54.40 | EXIT_EMA400 | -0.84 |
| BUY | 2025-09-30 09:15:00 | 56.06 | 2025-10-14 14:15:00 | 54.97 | EXIT_EMA400 | -1.09 |
| BUY | 2025-10-03 11:15:00 | 56.79 | 2025-10-14 14:15:00 | 54.97 | EXIT_EMA400 | -1.82 |
