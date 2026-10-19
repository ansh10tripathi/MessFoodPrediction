# Project guidance

- Keep the application modular: data handling, feature preparation, training, analytics, and UI belong in separate modules.
- Label generated development records as synthetic in files, UI, documentation, and reports. Never describe them as measured university results.
- Keep uploaded/real data in `data/real/`; do not overwrite an original upload or silently replace it with generated data.
- Prevent target leakage: future-consumption predictions cannot use actual attendance, preparation, consumption, leftover, or waste measurements.
- Use chronological splits for time-dependent model evaluation.
- Preserve the food-accounting identity `prepared_kg = consumed_kg + edible_leftover_kg + discarded_waste_kg`. Validation may use an absolute tolerance of 1e-6 kg for CSV floating-point round trips.
- Do not claim successful tests or model results without running them and reporting their actual outcome.
- Prefer standard-library functionality for data generation and validation when practical; keep generated data and model artifacts separate from source.
