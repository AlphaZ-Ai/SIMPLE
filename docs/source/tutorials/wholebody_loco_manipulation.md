# Wholebody Loco-manipulation

End-to-end quick start: download a trained Psi-0 checkpoint, serve it, and
evaluate it on the whole-body carry-box task — producing evaluation videos.

Prerequisite: the SONIC whole-body stack must be installed
(`bash scripts/setup_teleop_wbc.sh`) — see [Setup](../sonic-wbc/setup.md).

## 1. Download the checkpoint

From your **Psi-0** workspace, with `PSI_HOME` set:

```bash
export RUN=sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223

hf download USC-PSI-Lab/psi-model \
  --include="psi0/simple-checkpoints/$RUN/*" \
  --local-dir=$PSI_HOME/.runs \
  --repo-type=model
```

This run directory ships a single checkpoint, `checkpoints/ckpt_40000/`, and
lands at `$PSI_HOME/.runs/psi0/simple-checkpoints/$RUN`.

## 2. Serve the policy

```bash
serve_psi0_sonic_http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step=40000 \
    --run-dir=.runs/psi0/simple-checkpoints/$RUN \
    --rtc-mode test_time \
    --action-exec-horizon=24
```

Keep this terminal open — the server must stay up for the whole evaluation.
Point `--run-dir` at wherever your checkpoint actually lives; the
[Evaluation](../sonic-wbc/evaluation.md) example uses `.runs/finetune/$RUN`.

## 3. Run the evaluation client

In a second terminal, from the SIMPLE repository root:

```bash
HEADLESS=0 bash scripts/run_eval_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/teleop_wbc/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0 \
  --host 127.0.0.1 \
  --port 8014 \
  --episode-start 1 \
  --num-episodes 1 \
  --dr-level 0 \
  --eval-dir /tmp/sonic-psi0-eval
```

`--port` must match the server. `HEADLESS=0` opens the desktop viewer and needs
a live X display; use `HEADLESS=1` without one — the videos are written either
way. The script starts the lockstep SONIC controller itself and tears it down on
exit.

## 4. Watch the results

Videos are written under `--eval-dir` as
`episode_<id>/<cam_name>_<success_flag>.mp4`:

```bash
mpv /tmp/sonic-psi0-eval/psi0/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0/episode_1/head_stereo_left_success.mp4
```

Success-rate statistics are printed on exit and logged to
`data/evals_decoupled_wbc/eval_stats.txt`.
