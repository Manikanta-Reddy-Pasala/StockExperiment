# NIFTY FIN SERVICE (^CNXFIN)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5015 bars)
- **Last close:** 27958.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 5000 pts (index)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** 778.80
- **Avg P&L per closed trade:** 111.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 19970.45 | 20307.27 | 20307.77 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 15:15:00 | 20475.50 | 20307.39 | 20307.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 13:15:00 | 20498.10 | 20328.45 | 20318.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 22353.10 | 22362.78 | 21868.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-02 09:15:00 | 22509.35 | 22221.78 | 21928.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-09 09:15:00 | 21979.55 | 22232.27 | 21981.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 25151.55 | 26063.54 | 26067.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 25042.35 | 26045.45 | 26058.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 25895.20 | 25804.89 | 25925.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 12:15:00 | 25712.95 | 25824.26 | 25921.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 25712.95 | 25824.26 | 25921.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 13:15:00 | 25667.80 | 25822.70 | 25920.46 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 25976.85 | 25810.90 | 25906.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 26655.40 | 25983.84 | 25983.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 26704.25 | 26046.20 | 26015.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 26004.45 | 26169.22 | 26087.26 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 25358.35 | 26015.80 | 26016.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 25128.30 | 25875.75 | 25942.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 15:15:00 | 24884.30 | 24882.08 | 25247.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 24437.40 | 24882.98 | 25228.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-04 12:15:00 | 25210.35 | 24883.02 | 25204.80 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 26323.25 | 25099.26 | 25095.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 26684.15 | 25115.03 | 25103.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 25272.30 | 25782.51 | 25497.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 25766.05 | 25752.29 | 25491.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 28530.00 | 28860.65 | 28503.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-29 10:15:00 | 28487.40 | 28847.70 | 28504.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 27500.20 | 28391.23 | 28395.47 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 28750.00 | 28337.45 | 28336.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 28848.35 | 28342.53 | 28338.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 28369.55 | 28415.87 | 28378.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-06 09:15:00 | 28731.65 | 28370.52 | 28359.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-17 12:15:00 | 29633.60 | 29962.51 | 29661.35 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 28296.05 | 30059.38 | 30062.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 27909.05 | 29835.91 | 29946.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 28172.60 | 27848.95 | 28670.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 27954.45 | 27888.61 | 28647.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 28923.25 | 27971.56 | 28628.54 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-02 09:15:00 | 22509.35 | 2024-02-09 09:15:00 | 21979.55 | EXIT_EMA400 | -529.80 |
| SELL | 2024-11-28 12:15:00 | 25712.95 | 2024-12-03 09:15:00 | 25976.85 | EXIT_EMA400 | -263.90 |
| SELL | 2024-11-28 13:15:00 | 25667.80 | 2024-12-03 09:15:00 | 25976.85 | EXIT_EMA400 | -309.05 |
| SELL | 2025-02-01 12:15:00 | 24437.40 | 2025-02-04 12:15:00 | 25210.35 | EXIT_EMA400 | -772.95 |
| BUY | 2025-04-08 09:15:00 | 25766.05 | 2025-07-29 10:15:00 | 28487.40 | EXIT_EMA400 | 2721.35 |
| BUY | 2025-10-06 09:15:00 | 28731.65 | 2025-12-17 12:15:00 | 29633.60 | EXIT_EMA400 | 901.95 |
| SELL | 2026-04-09 13:15:00 | 27954.45 | 2026-04-15 09:15:00 | 28923.25 | EXIT_EMA400 | -968.80 |
