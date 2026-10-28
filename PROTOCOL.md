# Fixed-title cross-language attention experiment

Protocol v1 with source amendment v2, 28 October 2026. The original version preceded pageview retrieval and fitting. [The source amendment](SOURCE_AMENDMENT_V2.md) preserves the initial failed coverage gate and adds four development topics before any fitting or forecast outcome summary. All model-selection and evaluation rules remain fixed.

## Decision, target and population

Test whether a fixed English-language history improves a related Spanish or Japanese page's forecast beyond its own past. The target is the Wikimedia API's daily `agent=user`, `all-access` view count for one fixed article title and UTC date. Counts are requests classified as user traffic, not unique people, nationality or language proficiency. The API can misclassify automated traffic.

The original twelve convenience-selected topics are Earth, Moon, Mars, Sun, Volcano, Earthquake, Tsunami, Artificial intelligence, Machine learning, Climate change, Olympic Games and Solar eclipse. Source amendment v2 adds Water, Photosynthesis, Computer and Music after metadata/coverage failure, without replacing or hiding an original topic. Current Wikidata sitelinks identify English, Spanish and Japanese titles. The fixed period is 1 January 2022 through 31 December 2025. No topic is selected using forecast results, popularity or an observed event peak. Missing or ambiguous title history is an explicit source limitation or exclusion, with every exclusion retained.

Counts measure fixed titles. Current mappings do not establish historical semantic equivalence, and redirect aliases are not silently aggregated. Record current page IDs, creation evidence and bounded title-move logs. If a required target title changed during the period, retain the evidence and exclude that topic-language pair before fitting. A current-title log search is not proof that every historical alias has been found; disclose that limit and avoid a concept-wide traffic claim. Require at least six development and three held topics in each target language, at least 500 fitting dates, 100 dates per validation fold and 300 final dates per pair; otherwise retain a source/coverage hold and do not fit the study.

## Information timing and separation

Daily prediction for date t uses only counts dated at most t-2. This is an engineering availability assumption; exact historical API publication times and revisions are unavailable. The experiment is retrospective. Calendar fields for t are known, while t-1 or t outcomes never enter features. Training labels stop two days before each fitting origin. A held-period feature may use earlier observations from that period once their two-day lag has elapsed, but model parameters remain fixed within a fold.

Only Spanish outcomes for eligible development topics select hyperparameters: the original Earth, Mars, Sun, Volcano, Earthquake, Artificial intelligence, Climate change and Solar eclipse, plus Water, Photosynthesis, Computer and Music from source amendment v2. Three forward validation folds cover January-April, May-August and September-December 2024. Training begins in 2022 and expands up to the two-day cutoff before each fold. No Japanese outcomes or the four held topics enter selection.

Moon, Tsunami, Machine learning and Olympic Games form a topic holdout for parameter-selection transfer. Japanese is the language holdout for the selection rule. These are holdouts from hyperparameter selection, not cold-start pages: each final model is fitted to its own page's pre-2025 history under the frozen rule. All eligible Spanish and Japanese pairs are scored on 2025. Report development/held topics and languages separately without choosing a test winner to suppress other comparisons.

## Features, baselines and finite search

Use log(1+views). Own-history features are values at lags 2, 7, 14 and 28 days, plus the mean log count over t-8 through t-2. Calendar features are six weekday indicators and one annual sine/cosine pair using the UTC date and a 365.25-day cycle. The feature definitions, not their empirical usefulness, motivate this compact seasonal design.

Compare three ridge families: own history; own history plus the same five English features; and own history plus English lags 7, 14, 21 and 35 and the mean over t-13 through t-7. This last family tests a slower English signal while preserving a full seven-day observation delay. Penalties are 0.01, 0.1, 1, 10 and 100 for each family: fifteen declared configurations. Each candidate fits each eligible Spanish development page in each fold; with all twelve development topics eligible under amendment v2, the search has 540 fits. Report actual eligibility and fit counts. All fold predictions, objective values, statuses, runtimes and failures remain inspectable.

All methods use the same eligible dates, beginning only once the maximum 35-day history is available, with every required English and target count present. Missing counts are never filled with zero. Each fit standardizes predictors using its training rows only and solves mean squared log error plus lambda times squared coefficient norm, with an unpenalized intercept. A deterministic positive-definite linear solve is used, with independent objective, gradient, normal-equation residual and Hessian checks. Report conditioning and boundary penalty choices. This certifies a finite fixed-design fitting objective, not globally optimal generalization.

Select one penalty within each family by lowest arithmetic mean of page-fold mean squared log error, so each development page/fold has equal influence. Break exact ties by smaller penalty. Report all three frozen families. Two simple comparators predict t-7 views and the arithmetic mean of t-7, t-14, t-21 and t-28 views. Negative predicted log counts are clipped to zero before exponentiation; retain both raw and transformed predictions. Do not change clipping or metrics after test inspection.

Freeze all selected penalties, preprocessing, coefficients, feature definitions, row identities and source hashes in a committed artifact before any 2025 model inference. Final refits use labels through 30 December 2024. The same frozen rule fits each eligible topic-language pair; no test selection or per-pair retuning occurs.

## Metrics, uncertainty and checks

Primary comparison is page-balanced mean squared log error and its square root, RMSLE. It gives multiplicative errors more comparable influence across pages and does not measure operational value by itself. Also report MAE, MSE and RMSE in view-count units; within-page seasonal MASE uses the fitting-period t-7 absolute-error scale. Show denominators, zero/missing handling and per-topic/language failures. No monetary or staffing saving is inferred.

Independently reconstruct every headline metric from exported row-level outcomes and predictions, without importing model metric helpers. Reconcile every API count and date key into the panel through a separate standard-library/SQL path. Check duplicates, missing days, title identity, temporal boundaries and exact feature lag arithmetic. Independently reconstruct standardized ridge solutions and predictions from exported coefficients and transforms.

For paired test differences, retain a 2,000-draw common calendar-month block bootstrap within each language and topic-role group, using seed 20261028. Resample common test months across all included pages, preserving simultaneous shocks. These are conditional intervals for twelve months in a fixed convenience panel, not independent-language, worldwide-reader or population confidence guarantees. Retain each method comparison rather than testing many combinations and reporting only significant ones.

Three scientific figures should explain the predictive tradeoff, paired language/topic differences and time-block sensitivity. Show units, baseline, full candidate evidence, uncertainty meaning and negative results. Review the rendered figures and manuscript. No new algorithm, exhaustive novelty, causal English-to-other-language influence or prospective deployment claim is planned.

## Release gates and budget

Require pass states for eligible source/rights, snapshot accounting, parameter-selection separation, retained trials, numerical fitting certificates, independent metrics, clean full replay, privacy, rendered figures/manuscript and exact reviewed GitHub CI. Preserve failures and label any test reuse after a correction retrospective. Use the program's three substantive review rounds, 45 active review minutes and 60 review calls; execution is separate. On a genuine persistent blocker, retain scoped incompleteness and progress unaffected studies.
