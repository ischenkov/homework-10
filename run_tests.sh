#!/usr/bin/env bash
# Runs deepeval tests with telemetry disabled.
# deepeval's telemetry module makes a network request at import time which
# causes the CLI to hang silently. DEEPEVAL_TELEMETRY_OPT_OUT must be set
# before the Python process starts (conftest.py is too late).

set -a
[ -f .env ] && source .env
set +a

export DEEPEVAL_TELEMETRY_OPT_OUT=YES

deepeval test run tests/ "$@"
