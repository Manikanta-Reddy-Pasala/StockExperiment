# Maruti Suzuki India Ltd. (MARUTI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 13320.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -668.98
- **Avg P&L per closed trade:** -111.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 12229.70 | 12528.11 | 12528.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 12166.00 | 12496.54 | 12512.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 12458.00 | 12381.76 | 12441.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 09:15:00 | 12292.15 | 12384.90 | 12439.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-29 14:15:00 | 12458.40 | 12383.14 | 12437.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 13292.00 | 12431.64 | 12427.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 13368.50 | 12440.96 | 12432.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 12595.65 | 12638.69 | 12543.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 13:15:00 | 12848.05 | 12627.93 | 12547.33 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-14 10:15:00 | 12569.05 | 12659.51 | 12570.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 11985.70 | 12502.64 | 12503.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 13:15:00 | 11966.30 | 12497.31 | 12500.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11331.15 | 11317.54 | 11640.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-10 11:15:00 | 11235.30 | 11313.82 | 11617.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-02 09:15:00 | 11431.00 | 11099.94 | 11360.28 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 11993.00 | 11525.51 | 11523.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 12108.55 | 11531.31 | 11526.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 12397.00 | 12436.13 | 12129.54 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 09:15:00 | 11624.80 | 12001.93 | 12003.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 11454.35 | 11891.92 | 11939.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11830.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-16 09:15:00 | 11678.00 | 11730.18 | 11830.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 11785.00 | 11720.89 | 11812.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 13:15:00 | 11839.00 | 11724.01 | 11812.16 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 12503.00 | 11872.23 | 11871.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 12563.00 | 11879.10 | 11874.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 12325.00 | 12346.75 | 12169.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 10:15:00 | 12471.00 | 12304.81 | 12191.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 12302.00 | 12357.61 | 12238.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 12320.00 | 12357.23 | 12238.68 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-30 13:15:00 | 12360.00 | 12530.37 | 12380.68 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.42 | 16031.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.22 | 16008.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13626.00 | 13304.14 | 14062.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 13180.00 | 13365.71 | 14019.53 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-29 09:15:00 | 12292.15 | 2024-08-29 14:15:00 | 12458.40 | EXIT_EMA400 | -166.25 |
| BUY | 2024-10-09 13:15:00 | 12848.05 | 2024-10-14 10:15:00 | 12569.05 | EXIT_EMA400 | -279.00 |
| SELL | 2024-12-10 11:15:00 | 11235.30 | 2025-01-02 09:15:00 | 11431.00 | EXIT_EMA400 | -195.70 |
| SELL | 2025-04-16 09:15:00 | 11678.00 | 2025-04-23 13:15:00 | 11839.00 | EXIT_EMA400 | -161.00 |
| BUY | 2025-06-13 10:15:00 | 12320.00 | 2025-06-16 11:15:00 | 12563.97 | TARGET | 243.97 |
| BUY | 2025-06-06 10:15:00 | 12471.00 | 2025-06-30 13:15:00 | 12360.00 | EXIT_EMA400 | -111.00 |
