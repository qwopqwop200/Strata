# Atomic compact/full tokenizer config compatibility

New worktree off75c7; old04d2 and75c7 source/freezes unchanged. Actual pack writer uses compactspecial_ids{bos,eos,pad}, while Tokenizer.extract's schema uses originalGGUF fullkeys. The reader now accepts either representation only when values match actual Atomic rawKV exactly; optionalgguf_special_ids must also match. Optionalvocab_size/n_merges/model, if present, must agree; absence is checked using exact fullvocab/merge equality. Originaladd_bos/eosfalse flags are mandatory, and any providedpre_pattern must be the actualJoyAI/DeepSeek3 sequence.

ActualAtomicheader source47d07d... + originalISTAtemplate CPU13tests PASS undercommonlock1.173s/maxRSS234500KiB. Existing tests plus actualcompact config/mirroredfullGGUFids andwrongpad0rejection pass. The provisional materializer now writes bothaddflags explicitly. No sharedpack/oldmodel/dlservice writes, noGPU/model execution. FullAtomicfile verification/pack readiness remainsfalse until parentproducerproof and downloaderALL_VERIFIED.

This is a reader compatibility change, not a tokenizer vocabulary/merge/ID/policy change. Final pack writer remainsagent1. New standaloneAtomic session will use this reader after remaining strictpackproof/lifetime/resource validation; no old825 profile/globaloldmodel import is restored.
