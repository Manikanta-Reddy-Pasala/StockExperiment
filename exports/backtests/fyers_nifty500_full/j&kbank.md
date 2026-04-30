# Jammu & Kashmir Bank Ltd. (J&KBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 129.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 13.82
- **Avg P&L per closed trade:** 2.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 13:15:00 | 114.07 | 96.53 | 96.51 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 91.75 | 96.75 | 96.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 91.25 | 96.42 | 96.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 96.60 | 95.82 | 96.26 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 101.15 | 96.68 | 96.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 103.41 | 96.87 | 96.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 102.99 | 103.33 | 101.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 104.55 | 103.21 | 101.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 106.10 | 110.14 | 107.31 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 103.70 | 105.70 | 105.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 103.20 | 105.63 | 105.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 10:15:00 | 103.45 | 102.70 | 103.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 100.73 | 103.30 | 103.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-30 09:15:00 | 103.74 | 102.98 | 103.67 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 105.05 | 104.05 | 104.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 105.51 | 104.06 | 104.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 105.55 | 105.55 | 104.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 107.09 | 105.32 | 104.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 105.88 | 106.35 | 105.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-21 14:15:00 | 105.11 | 106.31 | 105.57 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 101.37 | 105.26 | 105.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 100.94 | 105.18 | 105.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 102.02 | 101.73 | 103.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 11:15:00 | 101.47 | 102.22 | 103.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-14 12:15:00 | 102.84 | 101.87 | 102.83 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 105.92 | 103.29 | 103.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 106.45 | 103.60 | 103.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 10:15:00 | 109.35 | 109.52 | 106.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-09 11:15:00 | 110.35 | 109.53 | 106.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 10:15:00 | 110.80 | 115.17 | 110.96 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-24 09:15:00 | 104.55 | 2025-06-30 09:15:00 | 114.29 | TARGET | 9.74 |
| SELL | 2025-09-26 09:15:00 | 100.73 | 2025-09-30 09:15:00 | 103.74 | EXIT_EMA400 | -3.01 |
| BUY | 2025-11-10 09:15:00 | 107.09 | 2025-11-21 14:15:00 | 105.11 | EXIT_EMA400 | -1.98 |
| SELL | 2026-01-09 11:15:00 | 101.47 | 2026-01-14 12:15:00 | 102.84 | EXIT_EMA400 | -1.37 |
| BUY | 2026-03-09 11:15:00 | 110.35 | 2026-03-10 14:15:00 | 120.79 | TARGET | 10.44 |
