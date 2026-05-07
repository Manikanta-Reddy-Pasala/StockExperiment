Profitable nifty50 stocks (180d, ₹200,000 capital, equal-weight ~₹3,774/symbol)
================================================================================
Date: 2026-05-08
Window: 180 days (6 months)
Universe: 53 NSE largecaps (NIFTY 50 + recent reconstitution)
Sustain: BUY=15m, SELL=75m
SL semantics: close-based (BTC trade rules v1.2)

CASE 1 — BUY plain (no filters)
  ADANIPORTS  +6.4%  +₹242   (1 leg)
  Final: ₹198,713 (P&L -₹1,287, ROI -0.64%)

CASE 2 — BUY HTF (HTF filter on)
  ADANIPORTS  +6.4%  +₹242   (1 leg)
  Final: ₹198,713 (P&L -₹1,287, ROI -0.64%)

CASE 3 — SELL plain (no filters)
  HCLTECH    +15.0%  +₹2,264 (4 legs)
  Final: ₹198,642 (P&L -₹1,358, ROI -0.68%)

CASE 4 — SELL all (HTF + slope filters)
  (no profitable)
  Final: ₹199,464 (P&L -₹536, ROI -0.27%)

NOTES
- 180d too short for strategy. Most symbols have 0 trades (no fresh
  EMA200×400 crossover in window). 45/53 symbols flat in BUY plain.
- Drawdown small only because trade activity is small.
- For meaningful evaluation use 720d (2yr).
