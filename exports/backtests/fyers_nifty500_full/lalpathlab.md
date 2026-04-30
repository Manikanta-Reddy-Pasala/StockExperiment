# Dr. Lal Path Labs Ltd. (LALPATHLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1376.90
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
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -61.50
- **Avg P&L per closed trade:** -20.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 1544.85 | 1616.82 | 1617.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 1530.65 | 1612.75 | 1615.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 1540.05 | 1535.44 | 1562.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 1526.80 | 1535.37 | 1562.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 12:15:00 | 1564.85 | 1535.93 | 1562.39 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 1391.55 | 1335.17 | 1334.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 1395.65 | 1336.34 | 1335.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1391.20 | 1391.93 | 1372.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 10:15:00 | 1417.15 | 1392.15 | 1373.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1405.20 | 1430.26 | 1404.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-23 14:15:00 | 1401.90 | 1429.98 | 1404.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1543.00 | 1577.25 | 1577.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1536.80 | 1572.93 | 1575.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-04 10:15:00 | 1573.90 | 1576.27 | 1576.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1573.90 | 1576.27 | 1576.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-04 11:15:00 | 1582.10 | 1576.33 | 1576.62 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 09:15:00 | 1526.80 | 2024-12-09 12:15:00 | 1564.85 | EXIT_EMA400 | -38.05 |
| BUY | 2025-06-03 10:15:00 | 1417.15 | 2025-06-23 14:15:00 | 1401.90 | EXIT_EMA400 | -15.25 |
| SELL | 2025-11-04 10:15:00 | 1573.90 | 2025-11-04 11:15:00 | 1582.10 | EXIT_EMA400 | -8.20 |
