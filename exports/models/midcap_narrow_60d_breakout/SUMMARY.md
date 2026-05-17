# midcap_narrow_60d_breakout — SUMMARY (ANGELONE EXCLUDED)

**Honest result with corp-action anomaly removed.** ANGELONE excluded from pseudo-midcap universe.

## Stock pick logic (plain English)

1. **Universe build**: take all 500 stocks in NSE Nifty 500
2. **Compute liquidity**: 20-day ADV per stock
3. **Rank by ADV**: sort descending
4. **Skip top-30** (large-caps, already covered by N100 models)
5. **Take next 100** = pseudo-midcap universe (ADV rank 31-130)
6. **Daily breakout scan**: for each day, find stocks with:
   - Today's close > **40-day high** (breakout)
   - Today's volume > **2× 20-day avg volume** (confirmation)
   - Today's close > **200-day SMA** (long-term uptrend filter)
7. **Pick top-1** by volume ratio (most-confirmed breakout)
8. **Hold** until: TARGET +100% / TRAIL -20% from peak (after +10%) / MAX_HOLD 90 trading days

**Unique filter**: this is the ONLY model that's event-driven (not calendar-driven). It buys when a breakout signal fires, sits in cash otherwise. SMA20-exit DISABLED (lets winners ride through dips).

| Key knob | Value |
|---|---|
| Universe | Pseudo-midcap (ADV rank 31-130 from N500) MINUS ANGELONE |
| Breakout | 40-day high |
| Volume confirm | ≥ 2× 20-day avg |
| Long-term filter | close > 200-day SMA |
| Position | max_concurrent=1 |
| **Rebalance period** | **Event-driven (daily scan, fires on breakout)** |
| Universe rebuild | Once (end-of-data snapshot, lookahead) |
| Exit | TARGET +100% / TRAIL -20% (after +10%) / MAX_HOLD 90d |
| Costs | 10 bps slippage + ₹20 brokerage + 0.10% STT |

## Headline result (3-year backtest, ₹10L start, ANGELONE excluded)

| Metric | Value |
|---|---:|
| Final NAV | **₹4,792,492** |
| Total return | **+379.25%** |
| **3-yr CAGR** | **+68.60%/yr** (below 80% threshold) |
| Max DD (cash NAV) | 17.83% |
| Trades | 12 |
| WR | 75.0% (9W / 3L) |

With ANGELONE (data anomaly) included: CAGR +337.62%, NAV ₹8.38 Cr, 13 trades, 92.3% WR — see model README + TRADE_LEDGER.md.

## Returns by NSE cap segment

| Cap segment | Trades | Wins | Losses | WR | Total PnL ₹ | Avg PnL/trade ₹ |
|---|---:|---:|---:|---:|---:|---:|
| **Large** | 4 | 2 | 2 | 50% | +1,699,565 | +424,891 |
| **Mid** | 5 | 5 | 0 | 100% | +2,198,826 | +439,765 |
| **Small** | 3 | 2 | 1 | 67% | -314,111 | -104,704 |

## Full trade ledger — every entry with price, invested ₹, exit, gain/loss

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | **Invested** | Exit Date | Exit ₹ | **PnL ₹** | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | MAZDOCK | **Large** | Nifty 100 | 2023-05-17 | 407.39 | 2,454 | ₹999,735 | 2023-07-12 | 865.06 | +1,120,997 | +112.56% | TARGET |
| 2 | INDIANB | **Mid** | Nifty Midcap 150 | 2023-07-13 | 325.32 | 6,519 | ₹2,120,761 | 2023-10-11 | 422.73 | +632,187 | +30.07% | MAX_HOLD |
| 3 | GMDCLTD | **Small** | Nifty Smallcap 250 | 2023-10-13 | 404.15 | 6,812 | ₹2,753,070 | 2024-01-11 | 466.38 | +420,710 | +15.51% | MAX_HOLD |
| 4 | CHENNPETRO | **Small** | Nifty Smallcap 250 | 2024-01-12 | 848.75 | 3,739 | ₹3,173,476 | 2024-04-12 | 894.90 | +169,212 | +5.54% | MAX_HOLD |
| 5 | HINDZINC | **Large** | Nifty 100 | 2024-04-15 | 425.42 | 7,858 | ₹3,342,950 | 2024-07-15 | 659.04 | +1,830,550 | +55.07% | MAX_HOLD |
| 6 | OFSS | **Mid** | Nifty Midcap 150 | 2024-07-16 | 10,960.95 | 471 | ₹5,162,607 | 2024-10-14 | 11,719.72 | +351,840 | +7.03% | MAX_HOLD |
| 7 | HDFCAMC | **Large** | Nifty 100 | 2024-10-16 | 2,282.28 | 2,420 | ₹5,523,118 | 2025-01-14 | 1,930.60 | -855,764 | -15.32% | MAX_HOLD |
| 8 | INDUSTOWER | **Mid** | Nifty Midcap 150 | 2025-01-21 | 374.27 | 12,476 | ₹4,669,393 | 2025-04-21 | 407.09 | +404,346 | +8.88% | MAX_HOLD |
| 9 | TATACONSUM | **Large** | Nifty 100 | 2025-04-25 | 1,161.36 | 4,368 | ₹5,072,820 | 2025-07-24 | 1,071.73 | -396,218 | -7.63% | MAX_HOLD |
| 10 | PAYTM | **Mid** | Nifty Midcap 150 | 2025-07-25 | 1,101.10 | 4,248 | ₹4,677,473 | 2025-10-23 | 1,282.82 | +766,460 | +16.62% | MAX_HOLD |
| 11 | SCI | **Small** | Nifty Smallcap 250 | 2025-10-24 | 248.25 | 21,930 | ₹5,444,122 | 2026-01-22 | 207.23 | -904,033 | -16.44% | MAX_HOLD |
| 12 | INDIANB | **Mid** | Nifty Midcap 150 | 2026-01-23 | 904.35 | 5,020 | ₹4,539,837 | 2026-04-23 | 914.04 | +43,993 | +1.17% | MAX_HOLD |

**Caveats**: Pseudo-midcap universe lookahead doing most of the work. Real Nifty Midcap 150 on same strategy = -18% CAGR.
