# Six-wave synthetic demonstration

**All employees, organizations, outcomes, and relationships in this package are synthetic. They are not real-world evidence about Gallup effects or any employer.**

Run from the repository root:

```powershell
python synthetic_data_v2/six_wave_demo/build.py
```

The build script calls the unchanged generator once with `long_main`, six waves, 5,000 employees per wave, seed 1, and `--full-data yes`. Its exact generator command is also recorded in `manifest.json`. It refuses to overwrite an existing package, so a second run must use a fresh checkout or a deliberately emptied package under separate supervision. Never run six independent one-wave simulations.

Upload only these files, in order, using the application's single-file survey uploader. Each file replaces the prior upload and includes all historical employee-wave rows:

1. `uploads/long_main_seed01_wave_01_asof.csv`
2. `uploads/long_main_seed01_wave_02_asof.csv`
3. `uploads/long_main_seed01_wave_03_asof.csv`
4. `uploads/long_main_seed01_wave_04_asof.csv`
5. `uploads/long_main_seed01_wave_05_asof.csv`
6. `uploads/long_main_seed01_wave_06_asof.csv`

The cutoff for snapshot N is the latest valid survey date in wave N. Turnover, absence, and performance values **and their dates** stay blank until their recorded availability date is on or before that cutoff. Current-wave outcomes are pending. The generator's prior-performance rating is masked in all uploads: it is sampled from a current-wave latent trait without an availability date proving that the rating was known at survey time. Its true value remains in the private answer key.

Only `uploads/` may be given to the app. **Never upload `private_answer_key/` or `generated_work/`.** The answer key includes hidden future outcomes, manager quality, true risk, event labels, and planted parameters. `manifest.json` records cutoff dates, row and outcome counts, file sizes, hashes, and ingestion results. `generated_work/` is the generator's intermediate output, retained for audit.

Expected progression: the first upload supports current survey evidence; later uploads add comparisons and, as dates permit, observed outcomes and outcome-dependent analysis. Prediction and recommendation display depends on the application's model and validation gates. A factor, recommendation, or prediction is not guaranteed to change every wave. Wave 1 may currently be classified as outcome-capable solely because blank outcome columns exist; that is an application issue for a later phase.
