# Atomic tokenizer/template compatibility

실제 Atomic pinned header-only tokenizer KV SHA47d07d2ab1f5a69bd9e43e731c232c9afa35ed6ab11c9bc32b48f537d453399a를 사용했습니다. vocab129280/merges127741, prejoyai-llm, BOS0/EOS1/PAD1, add_bos/eos false를 보존합니다. header-only probe이며 전체 파일 무결성/모델준비 PASS가 아닙니다.

원 Strata Tokenizer는 pre가deepseek로 시작하지 않는joyai를Qwen35로fallback했습니다. pinned llama3cf03257/src/llama-vocab.cpp319..327에서 JoyAI/DeepSeek3/Hunyuan/HY_V4가 동일3pattern sequence를 사용함을 확인하여 exactalias를 연결했습니다. extract pre_pattern metadata도해당sequence를기록합니다. 원DeepSeek/Qwen branch는그대로입니다.

상위원ISTA a419... chat_template.jinja Gitblob8127f8d9ceaf293b3d7bc75121d279f8887f8479/SHA4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99를사용합니다. 렌더링은actual serve.frontend.ChatTemplate/literal_tags/mark_think_literals/unmark_think_literals로하고 thinking_mode=thinking/reasoning_effort62를명시합니다. DSML fallback으로다른template를대체하지않는지도검사합니다. 반복image block은원placeholder를보존하되 실제274span/vision tower는agent2의실경로가담당합니다.

CPU12actualheader/template tests PASS: commonabsolute flock/STOP/PDEATHSIG/lockFDthroughchild/ownedreap0/sourceprepost,0.973s/maxRSS179676KiB/GPU0/model0. 실제regex grouping, Unicode roundtrip/special IDs, effort62, control문자plainmask, 반복imageplaceholder, tamperedassets/옛provenance/incomplete download 거절을확인했습니다.

최종pack/tokenizer writer는agent1입니다. 제materialize_header_tokenizer.py는자체임시/사설CPU검증용export만하며sharedmodel이나최종pack에는쓰지않습니다. AtomicFrontend.render_messages -> text,ids,input_ids_sha256,asset_contract_sha256,image_placeholder_positions,vision_expansion_pending,model_readyfalse. encode_plain/decode도제공합니다. 전체downloadALL_VERIFIED receipt와실pack/build/modelguards는별도필수입니다. 원825oldAbiray profile을자동복사하지않으며새Atomic state/profile은fresh_absent로시작합니다. 별도engine/API text/code/cancel/1M/default regression driver를이어서연결합니다.
