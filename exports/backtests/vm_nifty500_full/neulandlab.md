# Neuland Laboratories Ltd. (NEULANDLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 15003.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 282.64
- **Avg P&L per closed trade:** 47.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 6074.20 | 6560.05 | 6560.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 11:15:00 | 6030.55 | 6549.98 | 6555.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 6448.65 | 6446.23 | 6498.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-06 12:15:00 | 6289.75 | 6442.89 | 6496.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-11 14:15:00 | 6500.45 | 6419.85 | 6478.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 7319.00 | 6517.24 | 6513.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 7829.45 | 6697.34 | 6608.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 11875.85 | 12184.08 | 11077.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-27 09:15:00 | 13061.50 | 12184.82 | 11115.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 12:15:00 | 11244.30 | 12216.31 | 11320.79 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 13750.30 | 14282.90 | 14284.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 13652.90 | 14276.63 | 14281.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 12:15:00 | 14166.05 | 14131.93 | 14201.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 14:15:00 | 13825.75 | 14127.75 | 14198.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 14082.50 | 14007.18 | 14127.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 15:15:00 | 14170.00 | 13996.98 | 14114.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 13274.00 | 12129.07 | 12126.77 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 11652.00 | 12164.99 | 12166.26 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 14761.00 | 12173.13 | 12162.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 14848.00 | 12317.67 | 12235.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 13238.00 | 13276.29 | 12854.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-20 11:15:00 | 13849.00 | 13170.61 | 12934.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 13192.00 | 13302.92 | 13043.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-29 10:15:00 | 13334.00 | 13293.46 | 13048.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-26 11:15:00 | 13957.00 | 14662.68 | 14051.87 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 15026.00 | 15951.47 | 15955.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 14646.00 | 15773.28 | 15860.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 14:15:00 | 14123.00 | 13950.79 | 14615.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 10:15:00 | 13519.00 | 13945.05 | 14602.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 11:15:00 | 13291.00 | 12588.73 | 13176.57 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 15295.00 | 13579.72 | 13572.31 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-06 12:15:00 | 6289.75 | 2024-06-11 14:15:00 | 6500.45 | EXIT_EMA400 | -210.70 |
| BUY | 2024-09-27 09:15:00 | 13061.50 | 2024-10-07 12:15:00 | 11244.30 | EXIT_EMA400 | -1817.20 |
| SELL | 2025-01-24 14:15:00 | 13825.75 | 2025-01-28 10:15:00 | 12706.32 | TARGET | 1119.43 |
| BUY | 2025-08-29 10:15:00 | 13334.00 | 2025-09-03 11:15:00 | 14189.12 | TARGET | 855.12 |
| BUY | 2025-08-20 11:15:00 | 13849.00 | 2025-09-26 11:15:00 | 13957.00 | EXIT_EMA400 | 108.00 |
| SELL | 2026-02-10 10:15:00 | 13519.00 | 2026-04-06 11:15:00 | 13291.00 | EXIT_EMA400 | 228.00 |
