About Dataset
Weekly living series of publicly posted LLM/AI API list prices from official vendor documentation pages only (not aggregators).

What this is

Append-only tidy panel: snapshot_date × provider × model_id × price_tier
snapshot_date: UTC calendar date labeling the snapshot (not a comparison axis)
Historical snapshot_date partitions are retained across versions (see CHANGELOG.md, SERIES.md, snapshot_index.csv)
Weekly Sunday 10:00 JST routine refreshes retrieved_at_iso and appends new dates (no duplicate routine)
Monthly Kaggle versions + CHANGELOG document schema/series history (T074)
USD per 1M tokens; batch/fast/long-context/peak-offpeak kept separate when listed
See EXCLUSIONS.md for intentional scope limits
Providers: OpenAI, Anthropic, Google Gemini, xAI, Mistral (listed only), Groq, DeepSeek, Kimi, TypeSafe (Jev)

License / terms (Other): independently compiled factual API list-price observations; no vendor endorsement; consult official pages before commercial reliance. See sources.md and EXCLUSIONS.md.

