# SHORT-Side Momentum on Nifty 100 — Risk Overlay Variants

_3-year backtest May 2023 → May 2026, N100 universe, monthly rebalance, short bottom-5 by 60d return, max-concurrent=5._

## Comparison

| Mode | 2023-24 ROI | 2024-25 ROI | 2025-26 ROI | **3-yr Avg** | **Max DD** | Trades |
|------|------------:|------------:|------------:|-------------:|-----------:|------:|
| short_baseline (no overlay) | -46.72% | -9.87% | -24.56% | **-27.05%** | 54.15% | 81 |
| **short_trail_5** ⭐ | -15.93% | **+16.82%** | -23.50% | **-7.54%** | 25.82% | 132 |
| short_trail_8 | -29.70% | +17.07% | -24.58% | -12.40% | 33.65% | 121 |
| short_quick_cover (7d) | -21.71% | +3.73% | -27.50% | -15.16% | 28.12% | 142 |
| short_negative_only | -30.48% | -10.78% | -0.47% | -13.91% | 39.66% | 68 |

## Best Variant: short_trail_5 (still loses)

5% trail SL on shorts (exit if stock rallies 5% from entry low):
- Halved 3-yr loss (-7.5% vs -27%)
- 2024-25: actually positive (+16.8%)
- 2023-24 + 2025-26: still negative (bull years)

But: even best overlay = NEGATIVE 3-yr avg. NO short variant achieves positive returns.

## Why Shorts Keep Failing

- 2023-25 = Indian bull market. Bottom-momentum stocks REBOUND faster than they fall further
- Mean reversion in N100 large-caps overwhelms anti-momentum edge
- Only sideways/bear years work (2024-25 had a brief correction phase)

## Real-World Blocker

Retail Indian equity SHORTING:
- ❌ NOT allowed in delivery (CNC)
- ✅ Allowed intraday MIS only — useless for monthly hold
- ✅ Via stock futures — but F&O list limits to ~180-210 stocks (subset of N100 yes, but margin 20-30%, roll cost, STT-on-sell-side adds 0.1%)
- Real-world fills slip badly on short side; backtest assumed ideal close fills

## Final Verdict

**Do NOT deploy shorts. Stay long-only.**

Best short variant = -7.54%/yr (still losing). Best long variant (BEST_RETURNS.md trailing_sl 8%) = **+51.14%/yr**.

Long-momentum beats short-momentum by **~59 pts/yr**.

For Indian retail in 2023-26 market regime, momentum works ONE direction: long only.

## Files (not committed yet)

```
tools/backtests/momrot_short_filtered.py
remote: /app/logs/momrot/short_filtered/_results/*.json
```
