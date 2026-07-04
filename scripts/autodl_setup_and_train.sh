#!/usr/bin/env bash
set -euo pipefail

bash scripts/autodl_prepare.sh
bash scripts/autodl_train.sh
