# Native DeepSeek JSON constraints and full vocabulary sampling

This local engine extension masks next-token logits before sampling. It does not add a schema prompt or repair
the result. Thinking runs until the model emits its normal `</think>` token. Only the answer is JSON constrained.
EOS is legal only after a complete JSON object. Incomplete output is an error, including a max-token cutoff.
Formatting whitespace is limited to four characters at each grammar boundary to prevent whitespace-only loops.
Structured streams retain the server's buffering policy. They deliver the exact generated JSON after a separate
parse and schema check. They do not rewrite whitespace or escapes.

`response_format.type` supports `json_object` and `json_schema` on Chat Completions and Responses. The native
capability must be advertised by the running engine. Other engines keep their existing dispatch.

Supported schema keywords are `type`, `properties`, `required`, `additionalProperties: false`, `items`,
`minItems`, `maxItems`, `enum`, and `const`. Primitive types are object, array, string, integer, number, boolean,
and null. Type arrays are supported. Array bounds must be integers from 0 through 256. Schema nesting is limited
to 64. `title`, `description`, `$schema`, `examples`, and `default` are annotations and do not constrain values.
Other keywords, including references, numeric bounds, patterns, formats, and combinators, return 400.

An object with declared properties must use `additionalProperties: false`. Properties are emitted in schema
declaration order; optional properties can be omitted. Duplicate property names cannot be generated in a closed
object. A plain `type: object` permits ordinary JSON objects. Strings accept raw valid UTF-8 and JSON escapes,
including tokens that split a Korean character across byte boundaries. Answer control tokens are banned.
Tools combined with a structured response remain unsupported.

`top_k: 0` means the entire actual vocabulary. A positive top_k is configurable up to the vocabulary size;
larger values keep the whole vocabulary. The chain remains penalties, top-k, top-p, min-p, temperature, and draw.
Temperature 0 is greedy. The native sampled path uses a portable host fallback over complete logit rows, with
double-precision normalization. The unconstrained greedy CUDA kernel remains available. This changes the seeded
draw order and is not a claim of byte-identical outputs or a speed improvement.

DSpark stays active. Verification uses the same masked target distribution, saved draft distribution, acceptance
ratio, and normalized positive residual over all token IDs. Sampled drafts keep their own grammar snapshots.
Deterministic drafts are point distributions and may be rejected by the target grammar. A rejected draft does
not advance live grammar state. Sampled draft selection does not use a token-dependent confidence gate, which
would condition and change the saved draft distribution. Dense draft storage uses the actual vocabulary count;
missing capacity is an error rather than a truncated distribution.

Run CPU checks by compiling `tests/dsv41/full_sampler_test.cpp` with `-Iinclude` and
`tests/dsv41/grammar_probe.cpp` with `-Isrc/dsv41`. Set `GRAMMAR_PROBE` to the second executable, then run
`python -m unittest serve.test_native_schema`. `tests/dsv41/full_sampler_gpu.cu` exercises the native device
staging and full draft/verification storage. Deployment artifacts record the actual tested backend and hardware.
