# Anand Rathi Wealth Ltd. (ANANDRATHI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3582.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 186.32
- **Avg P&L per closed trade:** 37.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 1981.63 | 1908.93 | 1908.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 12:15:00 | 1993.40 | 1910.50 | 1909.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 1912.58 | 1915.80 | 1912.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 15:15:00 | 1950.00 | 1915.89 | 1912.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1950.00 | 1915.89 | 1912.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-20 10:15:00 | 1907.00 | 1915.79 | 1912.42 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 1888.75 | 2022.23 | 2022.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 1883.60 | 2000.02 | 2009.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 1908.68 | 1901.45 | 1948.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 10:15:00 | 1864.40 | 1899.90 | 1944.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-20 09:15:00 | 1929.98 | 1868.54 | 1914.12 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1880.00 | 1813.78 | 1813.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1892.90 | 1814.57 | 1814.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 2876.80 | 2893.88 | 2709.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-06 11:15:00 | 2919.90 | 2877.75 | 2739.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 14:15:00 | 2945.00 | 3052.15 | 2955.38 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 2956.40 | 2986.92 | 2987.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 2942.00 | 2986.48 | 2986.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 2954.90 | 2979.46 | 2982.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2954.90 | 2979.46 | 2982.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-13 12:15:00 | 2989.70 | 2979.42 | 2982.60 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 3033.90 | 2985.76 | 2985.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 3055.00 | 2990.67 | 2988.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 3064.00 | 3066.80 | 3034.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-01 10:15:00 | 3129.10 | 3037.10 | 3027.57 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-19 15:15:00 | 1950.00 | 2024-09-20 10:15:00 | 1907.00 | EXIT_EMA400 | -43.00 |
| SELL | 2025-02-07 10:15:00 | 1864.40 | 2025-02-20 09:15:00 | 1929.98 | EXIT_EMA400 | -65.58 |
| BUY | 2025-10-06 11:15:00 | 2919.90 | 2025-11-18 14:15:00 | 2945.00 | EXIT_EMA400 | 25.10 |
| SELL | 2026-02-13 09:15:00 | 2954.90 | 2026-02-13 12:15:00 | 2989.70 | EXIT_EMA400 | -34.80 |
| BUY | 2026-04-01 10:15:00 | 3129.10 | 2026-04-09 09:15:00 | 3433.70 | TARGET | 304.60 |
