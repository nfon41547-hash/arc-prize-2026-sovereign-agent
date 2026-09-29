Push a NEW VERSION of the team runtime dataset (owner pushes):
  cd dataset_patch/arc3sdk-runtime-v32-hardening
  kaggle datasets version -p . -m "v4-servo-1 Duck-servo fusion (interceptor+breakers+transfer+codex+CASS), fail-open"
If the dataset does not exist yet under your account:
  kaggle datasets create -p .
Then re-run the notebook: expect log line 'vendor-arc3sdk ok' AND
cell 15 'v32-hardening-1 active=True' (no ModuleNotFoundError).
