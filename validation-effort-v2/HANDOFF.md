# Atomic API effort compatibility follow-up

New worktree off04d2; old frozen04d2 tree/files unchanged. Changed runtime Python files: serve/frontend.py,serve/server.py only. CUDA/backend model numerical code unchanged.

DeepSeek-family API canonical ASCII strings1..100 now normalize to integer effort, including request"62". Signed/padded/decimal/Unicode-digit/out-of-range strings remain rejected. Other families retain old parsing. Strata's existingmedium62 alias and budget-to-medium path normalize to62 because originalISTA template accepts low/high/max/numeric, not literalmedium. Actual original-template enable_thinkingtrue and thinking_mode'thinking' produce identical effort62 prompts.

Actual config/shared-default admission also rejected integer62 before this patch. clean_shared_defaults gains explicit optionalfamily argument; onlydeepseek41 accepts canonical1..100 int/string. Startup config and shared-setting/readback paths passsvc.family. Qwen/default-family old restrictions remain. API requests with no effort now consume cfg62 and emit the same actual prompt62.

Validation: CPU-v1 initial10 new tests and original-api-v1 original2 passed beforeconfig extension; preserved. Final CPU-v2 12 actualAtomicheader/ISTAtemplate HTTP+Service tests PASS1.479s/RSS155796KiB; original-api-v2 existing2 API/Anthropic budget tests PASS1.423s/RSS78640KiB. Commonabsolute lock/STOP/FD/PDEATHSIG/ownedreap0 and sourceprepost allPASS. ActualService IDs equal AtomicFrontend IDs, literalcontrol masking retained, realHTTP returns200 forstring62 and defaultcfg62, invalid101 returns400. These use CPU mock engines, not actualnewmodel/vision/1M/default acceptance.

Parent API configuration must explicitly set reasoning_effort62,reasoning_budget_tokens0,answer_roomfalse,repeat_stop_tokens0,reasoning_loop_recoveryfalse,reasoning_close_retryfalse,anthropic_thinkingmodel to avoid implicit thought shortening/wrap/retry during naturalEOS qualification. Vision/defaultstate is owned byagent2/parent. No actualconfig/sourceintegration/default pointer mutation performed here.
