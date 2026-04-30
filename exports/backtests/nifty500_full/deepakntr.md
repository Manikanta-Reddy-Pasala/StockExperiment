# Deepak Nitrite Ltd. (DEEPAKNTR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1736.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 647.16
- **Avg P&L per closed trade:** 215.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 2214.60 | 2061.69 | 2061.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 2243.95 | 2072.46 | 2066.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 14:15:00 | 2183.65 | 2187.26 | 2139.96 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 2025.85 | 2120.42 | 2120.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 2011.90 | 2119.34 | 2120.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 2074.00 | 2060.46 | 2085.95 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 2201.75 | 2101.04 | 2100.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 14:15:00 | 2208.00 | 2103.09 | 2102.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 2211.20 | 2211.94 | 2170.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 09:15:00 | 2263.55 | 2212.45 | 2171.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 2287.25 | 2373.86 | 2298.69 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 2202.55 | 2275.12 | 2275.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 12:15:00 | 2195.95 | 2274.33 | 2274.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 10:15:00 | 2186.70 | 2177.74 | 2212.34 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 2316.85 | 2230.72 | 2230.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 2329.85 | 2233.85 | 2232.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 2387.00 | 2410.65 | 2349.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-19 10:15:00 | 2460.95 | 2342.03 | 2329.71 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-19 11:15:00 | 2825.00 | 2914.68 | 2841.98 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 2654.20 | 2830.02 | 2830.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2632.05 | 2797.58 | 2813.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 2774.60 | 2773.79 | 2799.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 11:15:00 | 2723.40 | 2778.53 | 2799.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2745.60 | 2709.77 | 2746.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 10:15:00 | 2749.00 | 2710.16 | 2746.78 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 1684.00 | 1537.83 | 1537.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1723.10 | 1542.54 | 1539.96 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 09:15:00 | 2263.55 | 2024-01-18 09:15:00 | 2287.25 | EXIT_EMA400 | 23.70 |
| BUY | 2024-06-19 10:15:00 | 2460.95 | 2024-07-22 13:15:00 | 2854.66 | TARGET | 393.71 |
| SELL | 2024-11-08 11:15:00 | 2723.40 | 2024-11-13 12:15:00 | 2493.66 | TARGET | 229.74 |
