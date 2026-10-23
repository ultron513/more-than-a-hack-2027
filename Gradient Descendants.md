# Ball Possession Detection — Technical Report

**Competition:** More Than a Hack 2027 — Phase 1: Ball Possession
**Repository:** https://github.com/ultron513/more-than-a-hack-2027
**Working submission file:** `evaluation/results/candidate_temporal_v4.csv`
**Report status:** Initial draft; review the development-process caveat below before final submission.

## 1. Summary

This project creates one possession label per tactical-video frame. The output CSV contains the columns `frame,label`, has 199,319 data rows indexed from frame 0 to frame 199,318, and uses the required labels: `H` (home team), `A` (away team), `D` (dead ball), and `X` (outside the active play intervals / no label assigned by this pipeline).

The current system is a deterministic, rule-based temporal model. It combines automatic player-and-ball tracking with automatic Smart Tagging intervals, constructs frame-level evidence for home possession, away possession, and dead ball, decodes each half with a Viterbi sequence model, and merges short interior runs during post-processing. It does not train a statistical model from manually labelled examples.

## 2. Input assets and their use

The program expects the competition's permitted data bundle under `data/tactical/`.

| Asset | Use in the current pipeline |
|---|---|
| `BAR-ATH_tactical_atd.txt` | Read frame-wise player and ball tracking; derive ball visibility, ball speed, player counts, and distances from the ball to the closest tracked player on each team. |
| `BAR-ATH_tactical_atd.xml` | Map tracking slots to the home and away teams. |
| `BAR-ATH_tactical_smart_tagging.xml` | Read automatically produced phase and set-piece intervals and turn them into frame-aligned supporting features. These annotations are treated as noisy evidence, not possession ground truth. |
| `BAR-ATH_tactical_homography.csv` | Check the number and alignment of homography frames during asset validation. The present inference model does not use homography to transform coordinates. |
| `BAR-ATH_tactical.mp4` | Verify that the required video asset is present. The present pipeline does not decode or inspect video pixels. |

The current algorithm does not use the broadcast video. The frame validator checks that the required tactical assets exist, checks that the two XML files parse, and checks alignment of the tracking and homography frame indices.

## 3. Feature construction

### Tracking features

The tracking parser reads each frame's player slots and ball record, using the metadata XML to map each slot to `H` or `A`. Normalized pitch coordinates are scaled using a pitch length of 105.3 metres and width of 68.0 metres before distances are calculated. For every frame, the extracted table records ball coordinates and speed when available, ball visibility, tracked-player counts by team, nearest-player distances by team, the closest team's identity, and the difference between the away and home nearest-player distances.

The extractor verifies frame alignment and requires exactly 199,319 rows. In the supplied run, the ball was visible in 113,488 frames (56.9%); this is a property of the automatic tracking export, not an accuracy measure for the possession model.

### Smart Tagging features

The Smart Tagging loader provides phase intervals and labels. These intervals are projected onto the tactical-video frame timeline and converted into signals for tagged phases by team, set pieces, and several set-piece types. The extractor processed 638 annotation intervals in the current run, including 59 set-piece intervals. Since phase annotations do not necessarily coincide with possession boundaries, they provide supporting evidence only.

Tracking and tagging features are merged by frame after confirming both tables span frame indices 0–199,318. The merged table contains 27 columns.

## 4. Possession inference

### H/A evidence

The model calculates a team-control logit primarily from the difference between the teams' nearest-player distances to the ball. The evidence is weakened when the nearest tracked player is relatively far away. A small adjustment is applied when a Smart Tagging phase is assigned to only one team. The logit is clipped and converted to a home-versus-away probability with a logistic function.

### Dead-ball evidence

A separate heuristic score estimates dead-ball likelihood. Its signals include missing ball observations, large distances to the ball, a large imbalance between the teams' nearest-player distances, low ball speed, set-piece intervals, and a decaying pulse at the start of a set-piece interval. These signals are uncertain proxies: tracking loss or a slow ball does not automatically imply a stoppage.

The three probabilities are combined into frame-wise emissions for `H`, `A`, and `D`. A Viterbi decoder then finds the most likely state sequence under a fixed transition matrix. The matrix favours continuity of the current team but permits turnovers and transitions to or from dead ball. Each half is decoded independently, preventing the sequence model from carrying possession across halftime.

The active intervals in the implementation are `(11185, 90652)` for the first half and `(110808, 190198)` for the second half. Frames outside these intervals are labelled `X`. The interval from frame 110808 up to, but not including, frame 113742 is assigned `D` as the second-half lead-in before kickoff, following the frame boundaries encoded in the implementation.

### Post-processing

Within each active half, interior runs shorter than 50 frames (2 seconds at 25 frames per second) are merged only if the labels on both sides agree. This suppresses brief isolated label switches while preserving a short run that lies at a true boundary between different neighbouring states. The current script reports 188 merged runs for the reproduced candidate.

## 5. Training data and supervision

No manually labelled training dataset was used, and no manual frame-by-frame possession labels were created for the supplied match. The inference rules are hand-designed heuristics over the automatic tracking and Smart Tagging signals. Smart Tagging is automatically generated annotation data and is not treated as verified possession ground truth.

### Development-process disclosure

During development, the public five-minute ground-truth sample supplied with the repository was used to inspect evaluation metrics. Several heuristic dead-ball weights, H/A transition probabilities, and the short-run post-processing threshold were changed after reviewing those results. The official competition rules describe the public sample as a pipeline-checking resource rather than a tuning set. Therefore, this development history raises a compliance concern that is not resolved by the fact that the full-match private ground truth was not used. This report records that history explicitly; the team should seek organizer guidance before treating this particular tuned candidate as a final compliant submission.

No private ground truth, third-party human-produced event labels for this match, or manually curated possession labels for this match were used in the described pipeline.

## 6. Evaluation and current result

The official local evaluator grades only the published sample's graded frames in this check. The sample covers 7,500 graded frames and contains 17 ground-truth possessions. It is too small to support a reliable estimate of performance over the complete match.

The current candidate's local public-sample results are:

| System | Public macroF1 | Public spellF1 |
|---|---:|---:|
| `candidate_temporal_v4` | 0.6666 | 0.3394 |
| `baseline_smooth125` | 0.5812 | 0.4242 |

The competition's primary metric is `spellF1`, so the candidate does **not** beat `baseline_smooth125` on this public sample. The candidate has a higher per-frame macroF1, but that does not make it the better-ranked system. The measured spellF1 bootstrap intervals overlap substantially; the small sample does not establish which system will be better on the full match. These measurements are a record of the present development state, not a claim of competitive full-match performance.

## 7. Reproduction

The pipeline was run on Windows using Python 3.14.5, NumPy 2.5.4, and pandas 3.1.0. Exact results on other environments have not yet been verified.

1. Place the permitted assets in `data/tactical/` using the original filenames expected by the scripts.
2. Create and activate a Python virtual environment.
3. Install the pinned pipeline dependencies:

   ```powershell
   python -m pip install -r requirements-pipeline.txt
   ```

4. Rebuild the feature tables, generate `candidate_temporal_v4.csv`, and validate its format:

   ```powershell
   python run_pipeline.py
   ```

5. Optionally, run the repository's public-sample evaluation as a diagnostic:

   ```powershell
   python run_pipeline.py --evaluate
   ```

The main output is `evaluation/results/candidate_temporal_v4.csv`. The evaluator and validator are part of the competition repository. The source scripts are intended to regenerate the output from the permitted data bundle, rather than requiring checked-in tracking data or derived feature tables.

## 8. Limitations and next improvements

- The current model is heuristic and has no learned or calibrated emission probabilities. Several thresholds and transition probabilities are fixed constants.
- Nearest-player distance is only an imperfect proxy for control. It can confuse contested balls, loose balls, tackles, interceptions, and situations where the ball carrier is not the nearest tracked player.
- The ball is missing from a substantial fraction of tracking frames, and tracking identities can be noisy. Missing observations can mislead both team-control and dead-ball heuristics.
- Smart Tagging intervals describe phases and set pieces, not exact possession boundaries, and can be wrong or overlap.
- The present pipeline does not use the homography values for feature construction and does not use either video feed to recover missing or ambiguous tracking.
- Fixed temporal transition settings can over-smooth real turnovers or still leave false short possessions. The 2-second cleanup rule may erase genuine short possessions.
- There is no independent full-match labelled validation available to the team, so full-match performance is unknown. Further modelling should follow the competition rules and avoid tuning on the public ground-truth sample.

Plausible future work includes better probabilistic modelling of contested and loose balls, explicit temporal features around likely turnovers, use of homography-derived features where warranted, and evaluation on permitted labelled data from other matches. Any improvement should remain fully inspectable and reproducible by the organizers.

## 9. Repository hygiene

The public repository must contain source code, dependency pins, and documentation only—not the licensed match bundle or frame-level derived tracking data. The current `.gitignore` excludes the local `data/` directory and generated result CSVs. The possession CSV and technical document are separate submission deliverables and should be named using the team's registered name when preparing the final handoff.
