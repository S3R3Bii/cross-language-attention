# Source coverage amendment before fitting

The first twelve-topic snapshot fails the prespecified coverage gate. It contains 217 retained responses and 51,768 count rows. The original gate, exclusions, parser and source snapshot remain in Git history and under `data/source/`. No model has been fitted, candidate score calculated or final forecast outcome summary examined.

Two changes address data fitness without changing the forecast objective, model families, penalties, time boundaries, holdout languages/topics or minimum coverage:

1. The move-log request intentionally covers 2022 through retrieval, but the first eligibility implementation treated a 2026 move as if it occurred in the 2022-2025 analysis interval. Filter exclusion events to that declared interval. Preserve later events in the source record and disclose current-title/alias limits.
2. Extend the convenience panel with four development topics: **Water, Photosynthesis, Computer and Music**. They were chosen as established non-person topics with three-language coverage, before requesting any of their pageview outcomes. Keep all original twelve topics and their exclusions. No replacement is selected by popularity, forecast accuracy or event peaks. The four originally held topics and Japanese language-selection holdout remain fixed.

The revised source snapshot is under `data/source-v2/`. Identical existing requests may reuse their exact retained bytes and original retrieval timestamps, with reuse lineage recorded. New requests get fresh hashes and timestamps. The resulting panel has at most sixteen topics, twelve for development and four held; if all development topics qualify, the finite search has 540 fits. Actual eligibility and fit counts must be reported. The minimum remains six development and three held topics per target language, 500 fitting dates, 100 dates per validation fold and 300 test dates per pair.

This amendment responds to metadata/coverage evidence before fitting. It does not turn the original failed gate into a pass, reconstruct all historical aliases or establish a representative sample of Wikipedia. The eventual evaluation remains a retrospective current-snapshot benchmark under a declared two-day availability lag.
