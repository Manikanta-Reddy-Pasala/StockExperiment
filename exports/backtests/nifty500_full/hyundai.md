# Hyundai Motor India Ltd. (HYUNDAI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-10-22 09:15:00 → 2026-04-30 15:15:00 (2609 bars)
- **Last close:** 1817.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 172.34
- **Avg P&L per closed trade:** 57.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 1823.30 | 1799.88 | 1799.83 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1742.80 | 1799.58 | 1799.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 1708.15 | 1798.67 | 1799.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 1721.15 | 1705.23 | 1741.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 11:15:00 | 1693.70 | 1709.31 | 1741.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 1746.45 | 1709.04 | 1739.34 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1830.40 | 1718.47 | 1717.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 1835.70 | 1719.64 | 1718.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 2059.70 | 2073.24 | 1996.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 2084.60 | 2072.72 | 1998.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-09 09:15:00 | 2419.30 | 2539.16 | 2423.47 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 2321.80 | 2384.37 | 2384.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 2314.50 | 2383.67 | 2384.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 2367.30 | 2367.19 | 2375.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-05 14:15:00 | 2309.50 | 2367.32 | 2374.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-07 09:15:00 | 2342.00 | 2313.17 | 2334.07 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 11:15:00 | 1693.70 | 2025-03-27 14:15:00 | 1746.45 | EXIT_EMA400 | -52.75 |
| BUY | 2025-07-29 12:15:00 | 2084.60 | 2025-08-18 09:15:00 | 2342.19 | TARGET | 257.59 |
| SELL | 2025-12-05 14:15:00 | 2309.50 | 2026-01-07 09:15:00 | 2342.00 | EXIT_EMA400 | -32.50 |
