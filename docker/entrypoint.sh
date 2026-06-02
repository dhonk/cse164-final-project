#!/usr/bin/env bash
# Dispatch on the first arg so `docker compose run cse164 <cmd>` picks a task.
# `notebook` is the default (see CMD in the Dockerfile / compose `command`).
#
# The task bodies below are intentionally stubbed: the Python pipeline
# (train/eval/submit flow) isn't finalized yet. Fill in each `exec ...` line
# once the corresponding entrypoint script exists, then drop the stub message.
set -euo pipefail

cmd="${1:-help}"; shift || true

stub() {
  echo "[entrypoint] '$1' is not wired up yet." >&2
  echo "[entrypoint] Define it in docker/entrypoint.sh once the pipeline is ready." >&2
  exit 1
}

case "$cmd" in
  help|"")
    echo "Tasks: notebook | train | evaluate | validate | shell"
    echo "Run JupyterLab with: docker compose up notebook"
    echo "Run a task with:      docker compose run --rm cse164 <task>"
    ;;

  notebook)
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
         --allow-root --ServerApp.token= --ServerApp.password=
    ;;

  train)    stub train      ;;  # TODO: exec python src/train.py "$@"
  evaluate) stub evaluate   ;;  # TODO: exec python src/evaluate.py "$@"
  validate) stub validate   ;;  # TODO: exec python starter/validate_submission_csv.py "$@"

  shell|bash)
    exec bash
    ;;

  *)
    # Fall through: run anything verbatim, e.g. `... nvidia-smi`.
    exec "$cmd" "$@"
    ;;
esac
