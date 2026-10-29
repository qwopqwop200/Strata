# Prefill router 동일 가중치 token2 재사용

기존6e/5b58 초기4367-prefill 진단 trace의 gemv_f32_kernel9회 합167.452ms를 근거로 선택했습니다. 해당 구간 커널합의6.975%이며 전체128K 또는 요청 latency 비율이 아닙니다. 원 진단의 정상QUIT 실패도 PASS로 바꾸지 않습니다. 실제 메타데이터 layers.0.ffn.gate.weight는384×5120 BF16(3,932,160B)이고 원f32()가 BF16을F32로 변환합니다. 실제chunk router는 Model::flow_layer의 gemv_f32(l.gate_w,c.x,l.n_exp,D,T,c.glog,1024,s)입니다. 기존ahead3개 서로 다른W를 묶는 후보와 달리 같은W를 토큰2개에 재사용합니다.

원소스1d 기준 kernels.cu 한 파일에만 defaultOFF DS4_PREFILL_ROUTER_REUSE exact1을 연결했습니다. CUDA/n384/k5120/ldy1024/T17..8192만 새커널입니다. T<=16·기존staticrouter·다른형상·HIP/Metal 원경로를 보존합니다. 원float4의4개 fmaf 순서, lane K순서, XOR warp_sum 순서와 output layout/padding은 token별 동일합니다. 출력/입력 별도scratch·barrier·새수명·Model상태 변경0. grid.y만T→ceil(T/2), 각warp2독립accumulator입니다.

작업space검토: c.h(8192*4*5120*4=640MiB), c.sg/c.su(각72MiB), c.x/c.acc(각160MiB)는 큰 버퍼이나 live range가 겹쳐 단순alias는 미확정입니다. attention slab1024는 이미 원launcher 설정입니다. 이 후보는 reserved-memory 변경0이며 다른owner reserve실험과 분리됩니다.

W는 F32 7.5MiB라 L2에 남을 수 있습니다. 절감은 DRAM보다 L1/L2 load/issue일 수 있고, register 증가·grid 감소·x로드 비용이 이득을 없앨 수 있습니다. 공유W의 로드수 절반을 실제 bandwidth 절반이나 전체gain으로 주장하지 않습니다.

한정성분 계획: compile1/GPU1,120초 lock 예산(-M20/compile60/run30 포함;정리3초), native sm121 O3. 기존runner의 절대Dlock/fullchild FD/PDEATHSIG/ownedPGID reap/STOP/dependency before-after/source·payload exitfailclosed 보존. 실제weight readonly3.9MiB range1개, tensor file/metadata 원본 변경0. 최대GPUalloc 약232MiB, host활성working 약232MiB와 비교출력64MiB로 합1GiB 미만 예상입니다. 원model/mainmap advice/globalcache 조작0.

실제GPU가 필요한 검증: T1/16 fallback,17 special(Inf/NaN payload/±0/subnormal),31,4367,8192 총6 whole-output bits+inactivepadding+guard 비교, 원weight/input bytes 불변, resource regs/shared 기록. 그뒤 정상입력actual4367/8192 각각4reps×ABBA=8 timing batch. 양쪽 warmup. 불일치1개면 즉시실패·timing0, firstdiff index/token/row/activeguard/oldnewhex 기록. 입력원본BF16→F32 bitshift는 원f32 변환과 같은정확표현입니다.

채택gate: 성분에서 두 실제T가 일관되게 이겨야만 부모fullbuild+같은후보binary OFF/ON meaningful4367 naturaltyped task의 prefill/TTFT/whole request/quality+retaineddecode/메모리IO를 검증합니다. 128K capability는 유지하며 필요후속 actual검증은 부모queue입니다. 성분 음성이면 모델확대0. once marker는 dispatch선택일 뿐 launch성공/완료/요청별빈도·gain 증명이 아닙니다.
