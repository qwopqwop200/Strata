# DSpark device taps 사설 후보

833 기반. DS4_DSPARK_DEVICE_TAPS 문자열 정확 `1`만 활성. CUDA/single GPU/serial/TP<=1/current5120×3, dump 둘 다 없는 경우만 983040B 전용 scratch. 원 host scratch는 유지. optional allocation OOM이면 host fallback, 다른 CUDA 오류는 기존 CK로 실패. OFF 추가 GPU allocation/clock/log 없음. 아직 전체 CUDA 빌드·모델 및 속도 검증 안 함.

실제 중복: target layers 37..39 FP32 hc_mean 결과를 pinned host에 내려 보낸 뒤 같은 값을 main_proj 직전에 H2D. 새로운 경로는 동일 kernel/launch에 output base/stride만 바꾸고 동일 mv job(ncol, xf_cs, y_cs, matrix)을 사용. bf16 변환은 추가하지 않는다. xe는 중간 expert scratch이므로 alias 금지; 별도 allocation이다. 확인된 host 소비는 seed input과 DSpark dump뿐이며 dump에서는 후보를 비활성화한다. prefill window/chunk와 warm_flow는 포인터 전달이 없어 기존 경로. runtime.patch로 수정 범위를 확인할 수 있다.

포인터는 verify/step stack→forward→serial tap→seed의 호출 인자다. request-valid flag/재사용 state가 없다. 성공 target forward의 기존 head sync 뒤 rollback·seed 순서는 원형. seed는 m+1개 수락 행만 투영한다. seed 투영 read와 다음 tap write는 같은 단일 stream FIFO라 host wait 없이 순서 보장. 실패한 forward는 seed하지 않으며 다음 호출은 모든 해당 tap 행을 다시 쓰고 사용한다. CPU cancel 요청 자체가 tap scratch를 읽거나 해제하지 않는다. destructor는 기존 d.allocs 해제. chunk feed/slot restore 뒤 최초 decode도 자기 새 tap을 쓴다.

CPU fixture 준비: 실제 tap/seed-input-loop/eligibility 발췌, queued CUDA double. T1/2/6/16 × kept1..T × interrupted tap0/1/2/3=100쌍의 host/device 입력 바이트 비교, canary/host unchanged/copy4→0, failed projection→다음 fresh call, eligibility fallback. 실제 verify verdict/RNG/rollback/publication suffix가 seed 인자 하나 외 byte-identical인지 source check. CPU double은 actual kernel/BLAS/CUDA 수치 증명이 아니다. 현재 prepare-only source checks PASS; compile/run은 부모 queue 승인 대기. 예정 공통 blocking flock + STOP + timeout60, g++ O2 한 번 compile/실행, 필요하면 ASan은 별도 승인. 생성 CPP 기준 약 <12MiB CPU scratch, 단일 compiler.

별도 실제 CUDA 성분 제안(아직 실행/컴파일0): 원 hc_mean archive 사용 T1/2/6/16, fp32 finite/tie/signedzero/extremes 입력 old d.tap→D2H→H2D와 직접 stride 출력 간 GPU 바이트/CPU reference 비교, canary + 동일 stream 재사용. math/mv 구현은 변경하지 않으므로 main_proj 입력 동일성이 핵심; 전체 모델 validation은 부모 전용.

측정 근거/한계: instrumented 128K invalid-quality trace의 1625target forward/4547counted rows에서 3D2H+1H2D = 6500 copy API/558735360B 왕복을 제거할 수 있는 범위. copy 비용 미분리, lifetime verify/seed timer를 순수 copy 시간으로 간주하지 않음. 0.94MiB 증가가 budget 경계 expert 배치를 바꿀 수 있으므로 실제 슬롯수 기록/검증 필요. 속도 개선·품질 PASS·채택 미주장. 기존 sampler 후보/정책 변경을 섞지 않은 833 단독 후보.
