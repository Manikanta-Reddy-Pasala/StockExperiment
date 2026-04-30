# Hyundai Motor India Ltd. (HYUNDAI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-10-22 09:15:00 → 2026-04-30 15:15:00 (2629 bars)
- **Last close:** 1828.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 169.69
- **Avg P&L per closed trade:** 56.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 10:15:00 | 1819.10 | 1797.94 | 1797.94 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 1787.00 | 1797.91 | 1797.92 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1832.95 | 1798.25 | 1798.10 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 1707.85 | 1797.22 | 1797.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 1702.70 | 1796.28 | 1797.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 1721.50 | 1705.02 | 1741.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 11:15:00 | 1693.70 | 1709.14 | 1740.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 1746.25 | 1708.90 | 1738.74 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 1830.10 | 1717.31 | 1717.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 1835.70 | 1719.60 | 1718.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 2059.60 | 2073.16 | 1996.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 2084.60 | 2072.64 | 1998.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-09 09:15:00 | 2419.30 | 2539.10 | 2423.40 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 2322.00 | 2384.35 | 2384.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 2314.50 | 2383.65 | 2384.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 09:15:00 | 2351.00 | 2368.49 | 2375.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2351.00 | 2368.49 | 2375.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-02 13:15:00 | 2386.70 | 2368.48 | 2375.55 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 11:15:00 | 1693.70 | 2025-03-27 14:15:00 | 1746.25 | EXIT_EMA400 | -52.55 |
| BUY | 2025-07-29 12:15:00 | 2084.60 | 2025-08-18 09:15:00 | 2342.54 | TARGET | 257.94 |
| SELL | 2025-12-02 09:15:00 | 2351.00 | 2025-12-02 13:15:00 | 2386.70 | EXIT_EMA400 | -35.70 |
