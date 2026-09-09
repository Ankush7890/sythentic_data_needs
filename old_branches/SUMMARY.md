# Archived instruction following experiment branches

This branch (`instruction_old`) was created on 2026-09-09 from `origin/main`
(`3b6113aa60e4`) plus **one commit** that folds 7 retired experiment
branches into `old_branches/<branch>/`. Nothing was merged, rebased or
filtered: each directory is the **exact tree of that branch's tip commit**
(added with `git read-tree --prefix=old_branches/<branch>/ <tip>`), so
every result, probe pickle, log, config, viewer and source file the branch
ever committed is here byte-for-byte. Every directory's tree hash was
checked equal to its source tip's tree hash before the old branches were
deleted.

## How the source tip was chosen

- **origin**: the local branch was absent, identical to, or behind
  `origin/<branch>`, so the remote tip was archived.
- **local (ahead of origin)**: the local branch had commits on top of the
  remote tip (remote ⊂ local), so the local tip was archived and no remote
  content is missing.
- `experiment_instruction_cloud_5` (instruction branch): local and remote had
  diverged, but the only content difference was an untracked-by-intent
  credential file (`kaggle/kaggle.json`) on the local side; the remote tip
  was archived and the credential was **not** carried over.

## History

The commit history of each old branch is not in this branch (this is one
commit), but it is not lost: every archived tip is also tagged
`archive/<branch>` and the tags were pushed, so
`git log archive/<branch>` / `git checkout archive/<branch>` still work.

## Concept assignment

Branches were assigned to a concept from their result/probe directory names
(`results_hu_harm_*` → human harm, `results_*hs*`/`highstakes` → high stakes,
`results_instructions_*` → instruction). `experiment17_cloud` also carries
`eval_sets/` for all three concepts but its probes and runs are human-harm,
so it lives with human harm.

## Branches in this archive

| branch | source | tip | tip date | commits past main-base | main-base | tree MB | key dirs changed vs base |
|---|---|---|---|---|---|---|---|
| `experiment_instruction_cloud_1` | origin | `5b83e2fe91aa` | 2026-08-16 | 50 | `e76e4fd0e192` | 42 | probes/instructions_gemma27b_gptoss, probes/instructions_gemma27b_nemotron, results_instructions_gemma27b_gptoss, results_instructions_gemma27b_nemotron, results_instructions_gemma27b_vintage, run_gemma27b_instructions_attackers.sh |
| `experiment_instruction_cloud_2` | origin | `4d3ea064cd58` | 2026-08-18 | 44 | `e833e1e0b8fb` | 168 | probes/instructions_gemma27b_ens10_gptoss, probes/instructions_gemma27b_ens10_nemotron, results_instructions_gemma27b_ens10_gptoss, results_instructions_gemma27b_ens10_nemotron, run_gemma27b_instructions_ens10_attackers.sh |
| `experiment_instruction_cloud_3` | origin | `3c9453508cdf` | 2026-08-20 | 55 | `67c9ddda2ecd` | 65 | analysis/ceiling, analysis/ensemble_fusion_speedup.md, analysis/novelty, probes/instructions_gemma27b_ens10dev_gptoss, probes/instructions_gemma27b_ens10dev_nemotron, results_instructions_gemma27b_ens10dev_gptoss, results_instructions_gemma27b_ens10dev_nemotron, run_gemma27b_instructions_ens10dev_attackers.sh |
| `experiment_instruction_cloud_4` | origin | `80f652eff277` | 2026-08-20 | 43 | `47ead49dab3d` | 41 | probes/instructions_gemma27b_xmemo_gptoss, probes/instructions_gemma27b_xmemo_view8_gptoss, results_instructions_gemma27b_xmemo_gptoss, results_instructions_gemma27b_xmemo_view8_gptoss, run_gemma27b_instructions_xmemo_arms.sh |
| `experiment_instruction_cloud_5` | origin | `1404c999a34e` | 2026-08-21 | 45 | `3b7ed0ef1d95` | 124 | probes/instructions_gemma27b_xmemodesc_gptoss, probes/instructions_gemma27b_xmemodesc_nemotron, results_instructions_gemma27b_xmemodesc_gptoss, results_instructions_gemma27b_xmemodesc_nemotron, run_gemma27b_instructions_xmemodesc_arms.sh |
| `experiment_instruction_cloud_6` | origin | `6911dc80bfe7` | 2026-08-22 | 44 | `3b7ed0ef1d95` | 114 | analysis/ceiling, analysis/novelty, analysis/SUMMARY.md, probes/instructions_gemma27b_xmemocat_gptoss, probes/instructions_gemma27b_xmemocat_nemotron, results_instructions_gemma27b_xmemocat_gptoss, results_instructions_gemma27b_xmemocat_nemotron, run_gemma27b_instructions_xmemocat_arms.sh |
| `experiment_instruction_cloud_7` | origin | `1a474c350f18` | 2026-08-27 | 49 | `815770e4f5d3` | 51 | probes/instructions_gemma27b_evaldesc_drift, probes/instructions_gemma27b_evaldesc_omission, probes/instructions_gemma27b_scopecheck_exp24, probes/instructions_gemma27b_scopecheck_exp24_armA_told, probes/instructions_gemma27b_scopecheck_exp24_cue_told, probes/instructions_gemma27b_scopecheck_exp24_cue_told_v2, probes/instructions_gemma27b_scopecheck_exp24_nocue_told, probes/instructions_gemma27b_scopecheck_exp24_nocue_told_v2 |
