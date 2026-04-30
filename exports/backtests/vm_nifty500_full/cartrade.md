# Cartrade Tech Ltd. (CARTRADE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1623.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -174.20
- **Avg P&L per closed trade:** -43.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 699.50 | 712.04 | 712.09 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 780.10 | 711.63 | 711.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 12:15:00 | 795.40 | 715.89 | 713.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 14:15:00 | 725.75 | 727.53 | 720.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-07 09:15:00 | 767.75 | 727.90 | 720.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-07 14:15:00 | 719.00 | 728.22 | 720.90 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 12:15:00 | 661.00 | 715.01 | 715.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 15:15:00 | 652.35 | 713.25 | 714.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 688.10 | 683.60 | 696.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-18 14:15:00 | 673.35 | 697.45 | 701.09 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-23 14:15:00 | 700.50 | 694.97 | 699.40 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 10:15:00 | 718.25 | 703.12 | 703.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 11:15:00 | 721.05 | 703.30 | 703.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 15:15:00 | 836.00 | 839.42 | 790.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-29 09:15:00 | 862.30 | 839.65 | 790.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 785.20 | 845.54 | 800.83 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 1551.30 | 1601.28 | 1601.47 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 1677.10 | 1602.00 | 1601.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1717.00 | 1614.84 | 1608.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1627.10 | 1629.12 | 1617.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 1638.50 | 1629.22 | 1617.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-24 12:15:00 | 1617.30 | 1629.21 | 1617.78 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2519.00 | 2765.31 | 2765.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 2443.40 | 2730.93 | 2748.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1866.00 | 1813.40 | 2016.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 14:15:00 | 1782.80 | 1819.23 | 1989.11 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-07 09:15:00 | 767.75 | 2024-03-07 14:15:00 | 719.00 | EXIT_EMA400 | -48.75 |
| SELL | 2024-04-18 14:15:00 | 673.35 | 2024-04-23 14:15:00 | 700.50 | EXIT_EMA400 | -27.15 |
| BUY | 2024-05-29 09:15:00 | 862.30 | 2024-06-04 11:15:00 | 785.20 | EXIT_EMA400 | -77.10 |
| BUY | 2025-06-24 09:15:00 | 1638.50 | 2025-06-24 12:15:00 | 1617.30 | EXIT_EMA400 | -21.20 |
