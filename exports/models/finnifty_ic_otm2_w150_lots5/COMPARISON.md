# finnifty_ic_otm2_w150_lots5 vs finnifty_ic_otm4_w300_lots5

What changed between the prior FinNifty IC export and this one, and why.

## TL;DR

The previous variant (`OTM 4% / wing 300 / 5 lots`) had a slightly higher headline return (+989 % vs +853 %) but **a single losing trade could blow 48 % of capital**. Two consecutive worst-cases wipes you.

The new variant (`OTM 2% / wing 150 / 5 lots`) caps single-trade loss at **17.5 %** — you can absorb 4–5 max-losses in a row and stay solvent. It also produced a **more stable year-over-year** return profile and runs into much **less execution slippage** because every leg sits in the liquid band.

## Side-by-side

| Metric | OTM4 / W300 (prior) | **OTM2 / W150 (new)** |
|---|---:|---:|
| OTM_PCT | 4.0 | **2.0** |
| WING_WIDTH | 300 pts | **150 pts** |
| LOTS | 5 | 5 |
| Slippage model | flat 1 % | **tiered by distance (1×-15×)** |
| Trades | 35 | 36 |
| Win rate | 77.1 % | 77.8 % |
| Avg/mo | +30.00 % | +25.85 % |
| Total return (3 yr) | +989.84 % | +853.13 % |
| **Max single-trade loss** | **48.2 % of capital** | **17.5 %** |
| Max drawdown | n/a (lower-bound only) | -32.7 % |
| Year-over-year stability | wide (172→325 %) | **tight (177→245 %)** |
| 86 % expiry-rate trades | 77 % | **86 %** |

## What's structurally different

### 1. Shorts much closer to the money

`OTM 2 %` puts the short call/put right at the edge of the high-IV band. Credit collected per unit risk is ~3-5× higher than `OTM 4 %`. Stronger theta engine.

### 2. Wings stay in the liquid band

Wings at `OTM + 150 pts` sit ~3-3.5 % OTM total. Still inside FinNifty's normal volume zone — order book depth survives. Compare prior variant: wings at OTM 4 + 300 pts = ~7 % OTM, premium often ₹2-5 with ₹2-3 spread = 30-80 % slippage on execution.

### 3. Backtest slippage model is **realistic**, not flat

Prior export assumed a flat 1 % slip on every leg. New backtest uses a tiered model:

| Strike distance from spot | Slip multiplier on base 1 % |
|---|---:|
| < 2 % OTM (near-ATM) | **1×** (1 %) |
| 2-3 % OTM | 2× |
| 3-4 % OTM | 4× |
| 4-6 % OTM | 8× |
| > 6 % OTM | **15×** (catastrophic) |

This realism reduces the headline CAGR but the new variant's wings still sit in the 1-2× band — so the headline holds. The old variant lost ~30 % CAGR when re-run with realistic slip (from 989 % flat → ~70 % realistic at the wings).

### 4. Defined-risk envelope is much tighter

Old: `(300 − credit) × 60 × 5 = ₹85-90k` worst case ≈ 45 % of capital.
New: `(150 − credit) × 60 × 5 = ₹35-40k` worst case ≈ 17.5 % of capital.

Cuts the catastrophic-tail in half.

### 5. Higher expiry-rate

86 % of new-variant trades go to expiry vs 77 % of old. Stops trigger less often because the short strikes pay enough premium to ride out small adverse moves. Lower exit-slippage exposure.

## Why this is the live config

| Criterion | Winner |
|---|---|
| Best raw P&L | OTM4/W300 |
| Best risk-adjusted P&L | **OTM2/W150** |
| Best execution-realistic P&L | **OTM2/W150** |
| Best year-over-year stability | **OTM2/W150** |
| Survives 5 max-losses without ruin | **OTM2/W150** |
| Promoted to live 2026-05-22 | **OTM2/W150** |

## What about going more aggressive later?

After 6 months of live confirmation with `OTM2/W150`, the path forward is to shift one or two lots into a more aggressive sleeve (`OTM3/W200/lots3` or `OTM4/W400/lots3`) and run them in parallel — not to swap variants wholesale. The current variant is the engine; aggressive sleeves are amplification you bolt on once the engine is proven on live capital.

## Code references

- Strategy logic: `tools/models/finnifty_ic_otm4_w300_lots5/_base_logic.py`
- Live signal wrapper: `tools/models/finnifty_ic_otm4_w300_lots5/live_signal.py`
- Backtest engine: `tools/models/finnifty_ic_otm4_w300_lots5/sweep.py`
- Multi-leg executor: `tools/live/fyers_executor_options.py`
- Depth gate / LIMIT-walk: `tools/live/option_depth_check.py`
- Promotion commit: `d44c1185`
