"""Paper-2 experiments package.

Streaming closed-form forecasting with the core solver:
  ridge warmup (closed-form) -> online RLS/Woodbury per sample -> LRU context
  -> episodic-Hopfield regime memory -> drift-adaptive forgetting.

Targets (see PLAN): OneNet/DSOF no-leakage protocol tables (ECL/ETT/Weather/Traffic),
Microprediction live board (colophon), chaotic dynamics (Lorenz96/KS, NARMA10-30).
"""