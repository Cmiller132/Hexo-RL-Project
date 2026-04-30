# Phase 04 Import Audit Command Plan

`rg` is required by the phase doc but currently fails in this workspace with `Access is denied`; use `git grep --untracked` as the fallback and record both commands.

Required audits:

```text
rg "architecture\\.startswith|startswith\\(\"global_\"\\)" Python/src/hexorl/inference
git grep --untracked -n "architecture\\.startswith\\|startswith(\"global_\"" -- Python/src/hexorl/inference
```

```text
rg "submit_.*global|submit_.*dense|submit_.*sparse|submit_.*pair" Python/src/hexorl/inference
git grep --untracked -n "submit_.*global\\|submit_.*dense\\|submit_.*sparse\\|submit_.*pair" -- Python/src/hexorl/inference
```

```text
rg "Queue\\.get\\(|Queue\\.put\\(|\\.join\\(" Python/src/hexorl/inference
git grep --untracked -n "Queue\\.get(\\|Queue\\.put(\\|\\.join(" -- Python/src/hexorl/inference
```

```text
rg "pair_head_present|pair_prior_mix" Python/src/hexorl/inference
git grep --untracked -n "pair_head_present\\|pair_prior_mix" -- Python/src/hexorl/inference
```
