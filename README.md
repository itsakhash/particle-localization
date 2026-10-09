# particle-localization

Monte Carlo localization (a particle filter) for a robot on a known 2D map, written from scratch with NumPy.

![demo](demo.png)

The robot starts completely lost, with a cloud of particles (guesses of x, y and heading) spread over the whole floor plan.
Each step the filter moves every guess with the robot's motion command, scores each guess by how well the range scan it would see
matches the scan the robot really sees, and resamples toward the good guesses. A few steps later the cloud collapses onto the robot.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_localization.py                       # animated demo
python run_localization.py --seed 7              # different start position
python run_localization.py --kidnap-at 100       # teleport the robot mid-run and watch it recover
python run_localization.py --save-png final.png  # or --save-gif demo.gif
python run_localization.py --trials 30           # no window: run 30 seeded trials, print the stats
```

Blue dots are particles, the green arrow is the true pose, the red arrow is the estimate. The right plot is position error over time.

## What is in it

- **Map and sensor:** occupancy grid, a vectorized ray caster (checked against a slow reference in the tests), and a simulated 24-beam, 5 m range sensor with Gaussian noise and random outlier readings.
- **Motion:** noisy speed/turn-rate model; the simulated robot never drives through walls.
- **Filter:** predict, weight, systematic resampling when the effective sample size drops, small post-resample jitter, and particles that drive into walls are killed.
- **Kidnapped-robot recovery:** the filter tracks a fast and a slow average of how well scans match; when the recent match collapses (the robot was moved) it re-spreads part of the cloud.
- **Metrics:** "converged" means the estimate stays within 0.3 m of the truth for 10 steps in a row *and* the cloud is tight, so a lucky guess from a wide cloud doesn't count.

## Results

Measured on a MacBook with 2000 particles, 0.2 s per step, a random start pose, and a filter that starts with no idea where the robot is.
Seeds 1000+ and 2000+ were not used for tuning.

| Test | Trials | Success | Median time to converge | Median final error |
|---|---|---|---|---|
| Global localization from a cold start | 30 | 30/30 | 8 steps (1.6 s) | 3.7 cm |
| Robot teleported at step 100 | 20 | 20/20 | 15 steps (3.0 s) | 3.7 cm |

Success means the estimate stays within 0.3 m of the true pose for 10 steps in a row. Reproduce with:

```bash
python run_localization.py --trials 30 --seed 1000
python run_localization.py --trials 20 --seed 2000 --kidnap-at 100 --steps 300
```

The demo map is deliberately lopsided: in a symmetric room, two different poses can look identical from the robot's sensor.
The robot's wandering policy is simple, so this tests the filter, not an exploration strategy.

## Tests

```bash
python -m unittest -v test_localization.py
```

Covers the map, ray caster, sensor noise, motion model, collision handling, every filter step (weighting, resampling, injection, circular mean of heading),
the metrics, and full end-to-end runs including recovery from being kidnapped.

## Files

| File | Purpose |
|---|---|
| `localization.py` | map, sensor, motion, particle filter, experiments |
| `run_localization.py` | animation, picture export, batch statistics |
| `test_localization.py` | unit and end-to-end tests |
