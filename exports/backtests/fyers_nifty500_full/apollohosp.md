# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7620.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -401.98
- **Avg P&L per closed trade:** -67.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.19 | 7070.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7026.36 | 7047.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 6876.55 | 6923.63 | 6980.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 6567.85 | 6351.12 | 6535.71 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 7032.50 | 6618.64 | 6618.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.41 | 6626.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.17 | 6770.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 6911.50 | 6852.82 | 6770.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6911.50 | 6852.82 | 6770.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 10:15:00 | 6926.00 | 6853.55 | 6771.31 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 6886.00 | 6937.59 | 6856.71 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-02 09:15:00 | 7065.00 | 6938.86 | 6857.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-03 09:15:00 | 6838.00 | 6938.33 | 6860.29 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.03 | 7636.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.50 | 7627.42 | 7632.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.88 | 7293.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 7017.00 | 7213.80 | 7289.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 7170.00 | 7052.36 | 7157.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7429.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-13 11:15:00 | 7598.00 | 7543.91 | 7430.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-16 13:15:00 | 7425.50 | 7540.31 | 7433.59 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 09:15:00 | 6876.55 | 2025-02-11 09:15:00 | 6564.53 | TARGET | 312.02 |
| BUY | 2025-05-12 09:15:00 | 6911.50 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -73.50 |
| BUY | 2025-05-12 10:15:00 | 6926.00 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -88.00 |
| BUY | 2025-06-02 09:15:00 | 7065.00 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -227.00 |
| SELL | 2026-01-20 09:15:00 | 7017.00 | 2026-02-06 11:15:00 | 7170.00 | EXIT_EMA400 | -153.00 |
| BUY | 2026-03-13 11:15:00 | 7598.00 | 2026-03-16 13:15:00 | 7425.50 | EXIT_EMA400 | -172.50 |
