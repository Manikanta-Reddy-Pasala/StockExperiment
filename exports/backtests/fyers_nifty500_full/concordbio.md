# Concord Biotech Ltd. (CONCORDBIO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1140.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 421.75
- **Avg P&L per closed trade:** 70.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 1799.25 | 2093.58 | 2094.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1716.25 | 2080.25 | 2087.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 1721.85 | 1720.89 | 1819.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1646.90 | 1719.70 | 1815.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-28 12:15:00 | 1640.70 | 1554.63 | 1629.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 2009.00 | 1684.35 | 1682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 2052.50 | 1688.01 | 1684.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1833.50 | 1852.35 | 1781.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-17 12:15:00 | 1921.00 | 1820.98 | 1793.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1843.40 | 1856.60 | 1821.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 1795.20 | 1854.28 | 1821.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1611.40 | 1795.14 | 1795.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1600.80 | 1789.62 | 1793.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 1714.60 | 1734.18 | 1757.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1391.80 | 1371.14 | 1425.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-08 10:15:00 | 1331.20 | 1371.50 | 1423.47 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1325.50 | 1258.71 | 1326.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-09 09:15:00 | 1297.80 | 1259.10 | 1326.15 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1260.00 | 1201.88 | 1269.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-02 15:15:00 | 1192.50 | 1206.02 | 1264.61 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-29 10:15:00 | 1142.90 | 1077.62 | 1128.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 1646.90 | 2025-05-28 12:15:00 | 1640.70 | EXIT_EMA400 | 6.20 |
| BUY | 2025-07-17 12:15:00 | 1921.00 | 2025-07-31 09:15:00 | 1795.20 | EXIT_EMA400 | -125.80 |
| SELL | 2025-08-26 09:15:00 | 1714.60 | 2025-09-09 14:15:00 | 1584.70 | TARGET | 129.90 |
| SELL | 2026-02-09 09:15:00 | 1297.80 | 2026-02-12 09:15:00 | 1212.75 | TARGET | 85.05 |
| SELL | 2026-01-08 10:15:00 | 1331.20 | 2026-03-23 12:15:00 | 1054.40 | TARGET | 276.80 |
| SELL | 2026-03-02 15:15:00 | 1192.50 | 2026-04-29 10:15:00 | 1142.90 | EXIT_EMA400 | 49.60 |
