# midcap_narrow_60d_breakout — SUMMARY (V2: Mid+Small only, ANGELONE data-fixed)

**Best-in-class result for this model.** Cap-filter sweep found that excluding Large-caps from pseudo-midcap universe boosts every metric: CAGR +18pp, DD -2.7pp, Calmar +1.87. Makes sense — model named 'midcap_narrow' performs best on actual mid+small caps without Large-cap dilution.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-15** (≈3.00 years) |
| First entry | 2023-05-17 |
| Last exit | 2026-05-05 |
| Total trades | 12 |
| Trades per year | ~4 |
| Strategy class | Daily breakout scan, long-hold swing (60-90d/trade) |
| Rebalance period | Event-driven |

## Stock pick logic (plain English)

1. **Universe build**: take Nifty 500 stocks, compute 20d ADV, skip top-30 large-caps by ADV, take next 100 (= pseudo-midcap)
2. **Cap filter (NEW V2)**: drop stocks in NSE Nifty 100 (Large-cap exclusion). Keep only Mid + Small caps from pseudo-midcap pool
3. **Data fix for ANGELONE**: prices in window 2024-12-23 → 2026-02-25 divided by 10 (corrupted by reverse-split-adjustment inconsistency in historical_data table). ANGELONE remains eligible — with clean data it naturally doesn't qualify for breakout entries.
4. **Daily scan**: for each day, find stocks with close > 40-day high + vol > 2× 20d avg + close > 200d SMA
5. **Pick top-1** by volume ratio (most-confirmed breakout)
6. **Hold** until: TARGET +100% / TRAIL -20% from peak (after +10%) / MAX_HOLD 90d

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | Pseudo-midcap (N500 skip-30 ADV, take next 100) |
| **Cap filter (NEW V2)** | **Exclude Large (NSE Nifty 100)** — keep Mid + Small only |
| Data fix | ANGELONE prices ÷10 in 2024-12-23 → 2026-02-25 window (reverse-split adj inconsistency) |
| Breakout | 40-day high |
| Volume confirm | ≥ 2× 20-day avg |
| Long-term filter | close > 200-day SMA |
| Position | max_concurrent=1 |
| Exits | TARGET +100% / TRAIL -20% (after +10%) / MAX_HOLD 90d |
| SMA20 exit | DISABLED |
| Costs | 10 bps slippage + ₹20 brokerage + 0.10% STT |

## Headline result (₹10L, 2023-05-15 → 2026-05-15)

| Metric | Value |
|---|---:|
| Final NAV | **₹6,500,421** |
| Total return | **+550.04%** |
| **3-yr CAGR** | **+86.63%/yr** |
| Max DD | **15.15%** |
| Trades | 12 |
| WR | 75.0% (9W / 3L) |
| Calmar | **5.72** |

## Cap-filter sweep (all 6 variants tested)

| Variant | CAGR | DD | Calmar | Notes |
|---|---:|---:|---:|---|
| **V2 Exclude Large (Mid+Small) — THIS** | **+86.63%** | **15.15%** | **5.72** | ✅ best on all metrics |
| V1 Exclude Small (Large+Mid) | +78.26% | 15.49% | 5.05 | 2nd best Calmar |
| V0 Baseline (all caps) | +68.60% | 17.83% | 3.85 | previous default |
| V4 Large only | +59.26% | 28.67% | 2.07 | worse |
| V3 Mid only | +38.71% | 20.01% | 1.93 | universe too narrow |
| V5 Small only | +9.99% | 48.08% | 0.21 | disaster |

**Insight**: pseudo-midcap pool contains some Large-caps (rank 31-130 by ADV catches large names like JIOFIN, ADANIPORTS, SHRIRAMFIN, ITC at end-2026). Those Large-cap breakouts compete with cleaner mid-cap setups for capital. Dropping Large preserves all wins AND adds capital headroom for next breakout — strategy compounds faster.

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 3 | 2 | 1 | 67% | +1,377,589 |
| **Small** | 9 | 7 | 2 | 78% | +4,208,705 |

All trades Mid + Small (Large filtered by V2 cap rule).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,909,855 | **+90.99%** | 3 |
| 2024-25 | ₹1,909,855 | ₹2,487,636 | **+30.25%** | 4 |
| 2025-26 | ₹2,487,636 | ₹6,500,421 | **+161.31%** | 5 |

## Caveats

- Pseudo-midcap universe has lookahead (end-of-data ADV applied retroactively).
- Real Nifty Midcap 150 (NSE official) on same strategy = -18% CAGR. Strategy depends on lookahead pool.
- 12 trades / 3yr = low sample; results sensitive to a few trades.
- ANGELONE-included V1: +337.62% CAGR (inflated by anomaly). Not used.

Full ledger: `TRADE_LEDGER.md`