# Archived high stakes experiment branches

This branch (`high_stakes_old`) was created on 2026-09-09 from `origin/main`
(`3b6113aa60e4`) plus **one commit** that folds 9 retired experiment
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
| `experiment6_cloud` | origin | `422abc328817` | 2026-07-28 | 35 | `348c7af789e0` | 36 | probes/hs_llama1b_deepseekv4pro_guidance, probes/hs_llama1b_deepseekv4pro_noguidance, results_hs_llama1b_deepseekv4pro_guidance, results_hs_llama1b_deepseekv4pro_noguidance, run_hs_guidance_ablation.sh |
| `experiment7_cloud` | origin | `10d3896da891` | 2026-07-29 | 46 | `348c7af789e0` | 40 | probes/hs_llama1b_gptoss120b_guidance, probes/hs_llama1b_gptoss120b_noguidance, results_hs_llama1b_gptoss120b_guidance, results_hs_llama1b_gptoss120b_noguidance, run_gptoss120b_guidance_ablation.sh |
| `experiment8_cloud` | origin | `a085e9d4db44` | 2026-08-03 | 199 | `348c7af789e0` | 45 | probes/hs_gemma27b_deepseekv4pro_noguidance, probes/hs_gemma27b_gptoss120b_noguidance, results_hs_gemma27b_deepseekv4pro_noguidance, results_hs_gemma27b_gptoss120b_noguidance, run_gemma27b_hs_attacker_ablation.sh |
| `experiment9_cloud` | origin | `8c26c0122010` | 2026-08-17 | 111 | `4dce11a411b3` | 44 | probes/hs_gemma27b_deepseekv4pro_batch, probes/hs_gemma27b_gptoss120b_batch, results_hs_gemma27b_batch_ablation, results_hs_gemma27b_deepseekv4pro_batch, results_hs_gemma27b_gptoss120b_batch, run_gemma27b_hs_attacker_ablation_batch.sh, run_vintage_hs_gemma27b.sh |
| `experiment12_cloud` | origin | `462401a3d757` | 2026-08-11 | 40 | `e76e4fd0e192` | 50 | probes/hs_gemma27b_gptoss120b_prompt, probes/hs_gemma27b_nemotron3ultra_batch, results_hs_gemma27b_gptoss120b_prompt, results_hs_gemma27b_nemotron3ultra_batch, run_gemma27b_hs_batch_vs_prompt.sh |
| `experiment13_cloud` | origin | `c2bb7930e470` | 2026-08-11 | 38 | `e76e4fd0e192` | 44 | probes/hs_gemma27b_gptoss_batch_guidance, probes/hs_gemma27b_gptoss_batch_itermemo, results_hs_gemma27b_gptoss_batch_guidance, results_hs_gemma27b_gptoss_batch_itermemo, run_gemma27b_hs_itermemo.sh |
| `experiment15_cloud` | origin | `8d87a5ea27ef` | 2026-08-12 | 64 | `e76e4fd0e192` | 67 | probes/hs_gemma27b_gptoss_batch_rounds10, probes/hs_gemma27b_gptoss_batch_target60, results_hs_gemma27b_gptoss_batch_rounds10, results_hs_gemma27b_gptoss_batch_target60, run_gemma27b_hs_itermemo.sh, run_gemma27b_hs_scaleup.sh |
| `experiment18_cloud` | origin | `23bf8d7d0ed5` | 2026-08-19 | 44 | `f4a23dbd9633` | 68 | probes/hs_gemma27b_deepseekv4pro_devval, probes/hs_gemma27b_gptoss120b_devval, results_hs_gemma27b_deepseekv4pro_devval, results_hs_gemma27b_gptoss120b_devval, run_gemma27b_hs_devval.sh |
| `experiment19_cloud` | origin | `69fb350ceb31` | 2026-08-19 | 90 | `67c9ddda2ecd` | 91 | analysis/redteam_space_ens3_fast, probes/hs_gemma27b_gptoss120b_ens3, results_hs_gemma27b_gptoss120b_ens3, run_gemma27b_hs_devval.sh, run_gemma27b_hs_ens3_one_arm.sh, run_gemma27b_hs_ens3.sh |
