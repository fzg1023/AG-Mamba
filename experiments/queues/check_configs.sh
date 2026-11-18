#!/bin/bash
for f in "$(dirname "$0")/../../baselines/configs/"*.yaml; do
  if grep -q angular_resample "$f"; then
    echo "$f: YES"
  else
    echo "$f: NO"
  fi
done
