# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 10267.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** 294.34
- **Avg P&L per closed trade:** 42.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 12:15:00 | 6984.00 | 7222.80 | 7222.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 6879.00 | 7213.54 | 7218.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 7060.00 | 7003.84 | 7089.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-26 09:15:00 | 6688.10 | 7002.20 | 7080.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 7013.00 | 6930.30 | 7021.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-07 09:15:00 | 7063.00 | 6935.72 | 7021.20 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 12:15:00 | 7415.55 | 7076.03 | 7074.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 15:15:00 | 7423.60 | 7105.13 | 7089.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 7695.00 | 7698.50 | 7492.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-03 14:15:00 | 8025.25 | 7718.80 | 7529.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 14:15:00 | 8371.05 | 8621.51 | 8384.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 7895.00 | 8292.24 | 8293.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 7822.00 | 8263.40 | 8278.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 8289.65 | 8220.55 | 8254.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-23 12:15:00 | 8177.10 | 8220.12 | 8254.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 8177.10 | 8220.12 | 8254.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-23 14:15:00 | 8100.35 | 8218.27 | 8253.10 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 12:15:00 | 8212.90 | 8217.31 | 8251.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-25 09:15:00 | 8163.50 | 8216.61 | 8250.72 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-29 09:15:00 | 8250.65 | 8211.14 | 8245.61 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 8431.95 | 8261.37 | 8260.72 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 8104.95 | 8260.95 | 8261.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 8089.85 | 8244.89 | 8252.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 8126.45 | 8111.33 | 8173.02 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 8860.15 | 8216.00 | 8215.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 9010.95 | 8401.45 | 8318.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 9315.95 | 9375.93 | 9035.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-07 11:15:00 | 9391.45 | 9357.27 | 9051.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-18 09:15:00 | 10180.20 | 10493.15 | 10207.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 12620.00 | 13595.46 | 13597.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 12473.00 | 13245.26 | 13363.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 11:15:00 | 12650.00 | 12590.59 | 12915.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 10:15:00 | 12497.00 | 12692.14 | 12900.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-06 11:15:00 | 12882.00 | 12586.33 | 12816.02 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-26 09:15:00 | 6688.10 | 2023-11-07 09:15:00 | 7063.00 | EXIT_EMA400 | -374.90 |
| BUY | 2024-01-03 14:15:00 | 8025.25 | 2024-03-13 14:15:00 | 8371.05 | EXIT_EMA400 | 345.80 |
| SELL | 2024-04-23 12:15:00 | 8177.10 | 2024-04-29 09:15:00 | 8250.65 | EXIT_EMA400 | -73.55 |
| SELL | 2024-04-23 14:15:00 | 8100.35 | 2024-04-29 09:15:00 | 8250.65 | EXIT_EMA400 | -150.30 |
| SELL | 2024-04-25 09:15:00 | 8163.50 | 2024-04-29 09:15:00 | 8250.65 | EXIT_EMA400 | -87.15 |
| BUY | 2024-08-07 11:15:00 | 9391.45 | 2024-09-02 12:15:00 | 10410.89 | TARGET | 1019.44 |
| SELL | 2025-10-30 10:15:00 | 12497.00 | 2025-11-06 11:15:00 | 12882.00 | EXIT_EMA400 | -385.00 |
