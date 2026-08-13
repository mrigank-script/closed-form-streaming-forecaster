"""Streaming closed-form forecasting experiments.

Closed-form ridge warmup -> per-sample online RLS (Woodbury) -> optional fixed
LRU context (S2), scored under the no-information-leakage protocol of the
online-forecasting literature.
"""
