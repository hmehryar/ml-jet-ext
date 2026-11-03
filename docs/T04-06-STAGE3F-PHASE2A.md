# T04-06 — Stage 3F.2A (read-only fixed80_v2 policy + migration audit)

This patch is deliberately **not** a training-policy switch. The eight existing
runs use immutable frozen workers/configurations, and four legacy Slurm attempts
may still be active. No existing run, registry, checkpoint, Slurm script, or
training loop is changed by this patch.

`train_utils/t04_fixed80_policy.py` defines and hashes the future fixed80_v2
policy. `tools/t04_06_protocol_audit.py` prints a read-only 120-run migration
inventory and blocks migration of submitted/running/resumed jobs **even if**
`sacct` reports success, until the old registry attempt has been reconciled.

Install (from the project root):

```bash
unzip -o T04-06_stage3f_phase2a_policy_audit.zip -d .
python -m pytest -q tests/test_t04_06_protocol_audit.py
python -m pytest -q tests/
git diff --check
python -m tools.t04_06_registry verify
python -m tools.t04_06_protocol_audit preview
```

Expected original registry remains untouched. The preview is intentionally not
an authorization to migrate or submit. Specifically, do **not** alter the 4
existing Slurm jobs, change `early_stopping` in frozen YAML, or mark the three
legacy early-stopped runs as incomplete in the original registry.

Next phase (3F.2B): create a new v2 plan/registry with distinct identities and
versioned configurations; wire the frozen worker's early-stop decision to the
new policy; verify checkpoint compatibility and support explicit cross-protocol
parent lineage. New v2 attempts must not be submitted until their parent v1
Slurm attempts have been fully reconciled and hashes verified.
