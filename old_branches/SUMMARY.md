# Archived human harm experiment branches

This branch (`human_harm_old`) was created on 2026-09-09 from `origin/main`
(`3b6113aa60e4`) plus **one commit** that folds 16 retired experiment
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
| `experiment1_cloud` | local(ahead of origin) | `7adc349de2a4` | 2026-07-23 | 0 | `7adc349de2a4` | 40 | (no results dirs; see diff vs base) |
| `experiment2_cloud` | origin | `e012edd0b5f5` | 2026-07-24 | 22 | `5b1c83938f12` | 62 | probes/llama70b50_gptoss120b_run2, results_hu_harm_llama70b50_gptoss120b_run2 |
| `experiment3_cloud` | origin | `f473af738d6e` | 2026-07-24 | 19 | `f97221b5f372` | 29 | probes/llama70b50_gptoss120b_run2, results_hu_harm_llama70b50_gptoss120b_run2 |
| `experiment4_cloud` | origin | `be7898a4b50c` | 2026-07-28 | 29 | `b221974ddb63` | 78 | probes/llama70b50_gpt51_memo, probes/llama70b50_gpt51_nomemo, results_hu_harm_llama70b50_gpt51_memo, results_hu_harm_llama70b50_gpt51_nomemo, run_gpt51_memo_ablation.sh |
| `experiment5_cloud` | origin | `f95346856f57` | 2026-07-28 | 32 | `b221974ddb63` | 34 | probes/llama70b50_deepseekv4_memo, probes/llama70b50_deepseekv4_nomemo, results_hu_harm_llama70b50_deepseekv4_memo, results_hu_harm_llama70b50_deepseekv4_nomemo, run_deepseekv4_memo_ablation.sh |
| `experiment10_cloud` | local(ahead of origin) | `e0c148cd36ac` | 2026-08-05 | 28 | `4dce11a411b3` | 42 | probes/hu_harm_llama1b_deepseekv4pro_batch, probes/hu_harm_llama1b_gptoss120b_batch, results_hu_harm_llama70b50_deepseekv4pro_batch, results_hu_harm_llama70b50_gptoss120b_batch, run_llama1b_hu_harm_attacker_ablation_batch.sh |
| `experiment11_cloud` | origin | `13593c32c5d2` | 2026-08-17 | 192 | `e76e4fd0e192` | 89 | probes/hu_harm_gemma27b_deepseekv4pro_batch, probes/hu_harm_gemma27b_gptoss120b_batch, results_hu_harm_gemma27b_batch_ablation, results_hu_harm_gemma27b_deepseekv4pro_batch, results_hu_harm_gemma27b_gptoss120b_batch, results_hu_harm_vintage_cross_eval, results_probe_versions, run_attribution_hu_harm_gemma27b.sh |
| `experiment16_cloud` | origin | `5824228a144b` | 2026-08-17 | 21 | `e833e1e0b8fb` | 122 | probes/hu_harm_gemma27b_deepseekv4pro_batch_ens10, probes/hu_harm_gemma27b_gptoss120b_batch_ens10, results_hu_harm_gemma27b_deepseekv4pro_batch_ens10, results_hu_harm_gemma27b_gptoss120b_batch_ens10, run_gemma27b_hu_harm_attacker_ablation_batch_ens10.sh |
| `experiment17_cloud` | origin | `ccdec2b9a075` | 2026-08-19 | 116 | `fd83258e189c` | 55 | probes/hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval, probes/hu_harm_gemma27b_gptoss120b_batch_ens10_devval, results_hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval, results_hu_harm_gemma27b_gptoss120b_batch_ens10_devval, run_gemma27b_hu_harm_attacker_ablation_batch_ens10_devval.sh, run_gptoss120b_run2.sh, run_instructions_llama70b50.sh |
| `experiment20_cloud` | origin | `061999dc0d28` | 2026-08-21 | 43 | `67c9ddda2ecd` | 58 | probes/hu_harm_gemma27b_gptoss120b_itermemo150, probes/hu_harm_gemma27b_gptoss120b_itermemo150_view8, results_hu_harm_gemma27b_gptoss120b_itermemo150, results_hu_harm_gemma27b_gptoss120b_itermemo150_view8, run_gemma27b_hu_harm_itermemo_ablation.sh |
| `experiment21_cloud` | origin | `b22631785cea` | 2026-08-21 | 39 | `3b7ed0ef1d95` | 49 | probes/hu_harm_gemma27b_deepseekv4pro_probedesc, probes/hu_harm_gemma27b_gptoss120b_probedesc, results_hu_harm_gemma27b_deepseekv4pro_probedesc, results_hu_harm_gemma27b_gptoss120b_probedesc, run_gemma27b_hu_harm_probedesc.sh |
| `experiment22_cloud` | origin | `fc8e440379cc` | 2026-08-24 | 48 | `3b7ed0ef1d95` | 52 | analysis/novelty, analysis/offdist, analysis/persistent, probes/hu_harm_gemma27b_deepseekv4pro_datadesc, probes/hu_harm_gemma27b_gptoss120b_datadesc, results_hu_harm_gemma27b_deepseekv4pro_datadesc, results_hu_harm_gemma27b_gptoss120b_datadesc, run_ceiling_exp22.sh |
| `experiment23_cloud` | origin | `61978c8cb13e` | 2026-08-24 | 99 | `815770e4f5d3` | 57 | probes/hu_harm_gemma27b_gptoss120b_s3_control, probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc, probes/hu_harm_gemma27b_gptoss120b_s3_itermemo150, results_hu_harm_gemma27b_gptoss120b_s3_control, results_hu_harm_gemma27b_gptoss120b_s3_evaldesc, results_hu_harm_gemma27b_gptoss120b_s3_itermemo150, run_gemma27b_hu_harm_memo_ladder.sh |
| `experiment24_cloud` | origin | `2e059ff3a9d2` | 2026-08-24 | 42 | `815770e4f5d3` | 37 | probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh, probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_refusal, results_hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh, results_hu_harm_gemma27b_gptoss120b_s3_evaldesc_refusal, run_gemma27b_hu_harm_evaldesc_arms.sh |
| `experiment25_gptoss_base_cloud` | origin | `8b2f87ee04fc` | 2026-09-01 | 67 | `3b6113aa60e4` | 44 | probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc, probes/hu_harm_gemma27b_gptoss120b_gptossbase_itermemo150, results_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc, results_hu_harm_gemma27b_gptoss120b_gptossbase_itermemo150, run_gemma27b_hu_harm_gptossbase_arms.sh |
| `experiment26_deepseek_cloud` | origin | `490aa8df4334` | 2026-09-02 | 70 | `3b6113aa60e4` | 45 | probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc, probes/hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150, results_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc, results_hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150, run_gemma27b_hu_harm_dsbase_arms.sh |
