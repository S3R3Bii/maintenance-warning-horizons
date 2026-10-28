# Remaining-life error versus an inspection decision

Declared 28 October 2026 after a source/README audit, before numeric cohort inspection or model fitting.

## Decision and publication scope

Which model improves remaining-life estimates and places genuinely near-failure synthetic engines in a limited inspection batch? Compare prediction error with the number of engines inside a declared maintenance horizon, rather than assuming lower RMSE means a better inspection decision.

Use the original NASA-linked C-MAPSS degradation simulation archive, not the currently unavailable simulator software. This is **simulation data throughout**, not measurements from an operating fleet. The source portal leaves licensing unspecified. Raw archive members, original README and original paper therefore remain local; public releases contain original code, exact hashes, retrieval/source links, derived predictions, figures and checks. No blanket source-data license or NASA endorsement is asserted. The official public source remains accessible through the retrieval script. Manuscript drafts remain private.

## Fixed cohorts and information boundary

Fit on FD001 training engines 1–70. Tune on engines 71–85. Calibrate uncertainty on engines 86–100. The official FD001 test engines form the main test; official FD002 test engines form a separate cross-condition stress test. Do not use FD002 training data or either test set for fitting, tuning or calibration. Source engine IDs are local to a partition; namespace all identities by file/partition. The source describes FD001 as one operating condition and FD002 as six, with the same nominal fault type. This is a dataset/regime stress comparison, not proof that every test condition is previously unseen.

For training, true remaining life is the engine's final training cycle minus the current cycle. For official tests it is the final observed cycle plus the provided remaining-life value minus the current cycle. Future failure information is a label only. Features never include final cycle, true remaining life or future sensor measurements.

After at least 20 observed cycles, features are current cycle, three operating settings, 21 current sensor values and each sensor's past-20 mean, population standard deviation and linear slope, including the current cycle. Do not use centered windows or whole-engine normalization. Validate the actual 26-column numeric schema; the source's prose ends with an inconsistent sensor-number example, so do not invent physical sensor names. Labels are uncapped remaining cycles. Predictions are clipped at zero; no upper clipping.

## Search and fitting

Fifteen configurations: a mean-training-failure-age minus current-age baseline; six random forests (160 trees, maximum depth 4/8/unlimited, minimum leaf 5/20); eight histogram-boosting regressors (depth2/3, learning rate .05/.1, L2 0/1, 100 iterations, minimum leaf20, early stopping disabled). Seed20261028. Forests use square-root feature subsampling, bootstrap sampling and two workers. Each fitting engine receives equal total sample weight. Trees use unscaled features; no pretrained fine-tuning occurs.

Select by mean per-engine MSE on tuning engines, ties by candidate name. Retain every configuration, outcome, runtime, status and validation prediction. Report engine-balanced MSE, RMSE and MAE as well as pooled window errors and the final-window errors. A finite grid is not a global optimum. Baseline settings are fixed, not selected using test data.

## Calibration and final evaluation

After selection, predict every eligible window of each of the 15 calibration engines. Its nonconformity score is the maximum absolute residual over its observed trajectory. For nominal90% simultaneous trajectory coverage, choose the order statistic at ceil((15+1)×.9)=15, i.e. the largest calibration-engine score. Use the same engine-max calibration procedure for the baseline. Intervals are [max(0,prediction−q), prediction+q]. The exchangeability assumption is explicit and cannot be established by a coverage score; FD002 stress need not satisfy it. Do not call these conditional or production guarantees.

Commit the selected model, fitting/tuning/calibration identities and radius before the test command. Evaluate on both protected test partitions without further tuning. Retain per-window predictions, labels, intervals and source identities; report both point coverage and whether every observed eligible window of each test engine is covered. Also report error/coverage by true remaining-life bands 0–30,31–90,>90 cycles, clearly as descriptive outcome strata.

## Fixed inspection batch

Each test engine contributes its **last available window** to one benchmark inspection batch. This is not a synchronized real fleet: no common calendar or commissioning dates are invented. Allocate floor(10%×number of engines) inspections by lowest predicted remaining life; ties use prediction rounded to8 decimal places and then local engine ID. Report the engines inside true horizons15/30/60 cycles among the selected batch, horizon recall, selection precision and the true lead-time distribution. The main horizon is30 cycles. Counts outside that horizon are not labeled economic false positives. Intervals provide uncertainty context, not an alternative post-test ranking rule.

Compare error, uncertainty width/coverage and inspection capture together. No invented maintenance cost, dollar savings, safety benefit or operational deployment is claimed. Use paired engine resampling for error differences and report its conditional scope; never resample overlapping windows as independent engines. No statistical test will manufacture evidence from a small number of inspection selections.

## Release gates

Raw hashes/schema/counts and source labels reconcile. Fit/tune/calibration/test identities are disjoint by source definition. Past-only window features receive independent spot checks; all prediction/error/coverage/inspection calculations and calibration order statistics are independently recomputed. Relevant contracts cover future-input independence, engine-boundary isolation, label exclusion, unit accounting and budget/tie behavior. Render figures and a private manuscript. Public outputs exclude raw source members, original source paper, credentials, private information and manuscript drafts. Retain failures and a current-version receipt; source or portability blockers keep the release incomplete.
