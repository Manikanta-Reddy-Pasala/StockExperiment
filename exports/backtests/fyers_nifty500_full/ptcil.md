# PTC Industries Ltd. (PTCIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 16140.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -4416.40
- **Avg P&L per closed trade:** -630.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 12024.15 | 13384.59 | 13386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 11700.00 | 13339.41 | 13363.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 12268.95 | 12110.05 | 12526.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 13:15:00 | 11600.75 | 12023.35 | 12383.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-23 12:15:00 | 12167.00 | 11788.82 | 12130.42 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 13994.00 | 12372.99 | 12370.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 14:15:00 | 14442.45 | 12393.58 | 12380.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 12:15:00 | 14794.30 | 14802.57 | 13862.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-22 14:15:00 | 14899.95 | 14803.61 | 13872.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 14129.40 | 14823.38 | 14014.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-29 14:15:00 | 13848.95 | 14785.88 | 14015.48 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 14:15:00 | 10197.95 | 13760.06 | 13762.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 10074.15 | 13346.06 | 13548.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 12504.55 | 12171.04 | 12810.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-11 13:15:00 | 11700.00 | 12185.90 | 12763.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 12647.45 | 12186.53 | 12714.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-17 14:15:00 | 12346.95 | 12200.27 | 12710.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 12650.05 | 12206.32 | 12708.55 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 12733.55 | 12237.95 | 12704.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 14445.00 | 13022.15 | 13020.33 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 13:15:00 | 12230.00 | 13242.32 | 13245.86 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 14786.00 | 13254.50 | 13248.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 14972.00 | 13552.97 | 13411.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 14439.00 | 14489.55 | 14033.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-10 09:15:00 | 14730.00 | 14484.56 | 14047.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 15:15:00 | 14273.00 | 14704.83 | 14280.89 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 13692.00 | 14464.81 | 14465.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 13445.00 | 14447.28 | 14456.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 14049.00 | 13980.30 | 14161.96 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 15231.00 | 14290.54 | 14287.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 15566.00 | 14471.75 | 14383.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 17898.00 | 17909.15 | 17153.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-23 15:15:00 | 18270.00 | 17767.34 | 17229.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 17586.00 | 17975.59 | 17514.69 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-07 15:15:00 | 17700.00 | 17972.84 | 17515.61 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-08 15:15:00 | 17524.00 | 17959.47 | 17524.58 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 17311.00 | 17741.88 | 17743.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 17173.00 | 17736.22 | 17740.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 16201.00 | 16190.61 | 16674.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-28 10:15:00 | 16043.00 | 16187.63 | 16654.49 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-06 13:15:00 | 11600.75 | 2024-12-23 12:15:00 | 12167.00 | EXIT_EMA400 | -566.25 |
| BUY | 2025-01-22 14:15:00 | 14899.95 | 2025-01-29 14:15:00 | 13848.95 | EXIT_EMA400 | -1051.00 |
| SELL | 2025-03-11 13:15:00 | 11700.00 | 2025-03-19 10:15:00 | 12733.55 | EXIT_EMA400 | -1033.55 |
| SELL | 2025-03-17 14:15:00 | 12346.95 | 2025-03-19 10:15:00 | 12733.55 | EXIT_EMA400 | -386.60 |
| BUY | 2025-06-10 09:15:00 | 14730.00 | 2025-06-19 15:15:00 | 14273.00 | EXIT_EMA400 | -457.00 |
| BUY | 2025-12-23 15:15:00 | 18270.00 | 2026-01-08 15:15:00 | 17524.00 | EXIT_EMA400 | -746.00 |
| BUY | 2026-01-07 15:15:00 | 17700.00 | 2026-01-08 15:15:00 | 17524.00 | EXIT_EMA400 | -176.00 |
