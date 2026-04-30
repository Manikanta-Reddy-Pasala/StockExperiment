# Acutaas Chemicals Ltd. (ACUTAAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2600.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 262.85
- **Avg P&L per closed trade:** 131.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 1098.00 | 1144.34 | 1144.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1075.80 | 1142.51 | 1143.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 1134.30 | 1129.75 | 1136.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-23 12:15:00 | 1095.40 | 1128.62 | 1135.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 1131.90 | 1120.87 | 1130.76 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1225.00 | 1133.82 | 1133.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 1256.00 | 1158.68 | 1148.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1408.60 | 1417.32 | 1348.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 12:15:00 | 1460.00 | 1402.55 | 1360.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 1621.80 | 1713.70 | 1633.62 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-06-23 12:15:00 | 1095.40 | 2025-06-27 09:15:00 | 1131.90 | EXIT_EMA400 | -36.50 |
| BUY | 2025-10-10 12:15:00 | 1460.00 | 2025-10-20 14:15:00 | 1759.35 | TARGET | 299.35 |
