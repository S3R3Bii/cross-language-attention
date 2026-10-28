# Does an earlier English signal improve a later language-edition forecast?

An editor allocating attention needs a forecast that improves on a page's own history. This study tests whether lagged English Wikipedia pageviews improve next-day Spanish and Japanese fixed-title forecasts. It compares error on the log scale with errors in pageviews, retaining failures by topic and language.

**Current state: source gate passes; no models fitted yet.** The revised sixteen-topic snapshot retains 289 response records and 69,300 daily count rows. Independent SQL/source checks reconcile every count, 48 title identities and all 32 candidate pair-date manifests. Twenty-six Spanish/Japanese pairs pass the unchanged coverage minimums: nine Spanish development/three held topics and ten Japanese development/four held topics. Six pair exclusions remain explicit. No forecast outcome summary has been examined.

The original twelve-topic snapshot and its failed gate are retained in [the source amendment evidence](results/corrections/source-v1-coverage-hold.json). [The pre-fit amendment](SOURCE_AMENDMENT_V2.md) corrects the title-move date interval and adds four development topics without changing model grids, metrics, time boundaries, holdouts or coverage minima.

The planned comparison uses current historical data, not historical API release vintages. A two-day input lag is a declared availability assumption. Results will therefore describe a retrospective benchmark and cannot establish real-time forecast performance, reader geography, language fluency or a causal direction of attention.

See [the fixed protocol](PROTOCOL.md) and [source and method context](SOURCES.md). All candidate trials, predictions, numerical fitting certificates, independent checks and limitations will be retained. Independent research manuscripts may accompany the reviewed GitHub package; formal-publication projects are outside this study.
