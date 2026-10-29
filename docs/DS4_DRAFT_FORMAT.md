# Draft Markov head format

`DS4_MARKOV_HEAD_Q8=0` loads the drafter's whole Markov head as F16 from the original model tensor.
`DS4_MARKOV_HEAD_Q8=1` uses the upstream Q8_0 in-memory form. With the variable unset, the upstream release
choice applies: Q8_0 in 0.2.7, or F16 with `DS4_DEFAULTS=026`. An explicit value wins in both cases.
This switch changes no model file and does not roll back other 0.2.7 defaults.

The verbose startup line `draft markov-head` records the whole-head format, the greedy path, the sampled path,
the override and the release defaults. With a split head group, the existing greedy path uses F16 shares;
sampled drafts use the whole head. The line describes the loaded paths and does not mean sampled drafting is enabled.
Hold this setting fixed when comparing adaptive verification with the old drafter. Q8_0 changes rounding;
no answer-equivalence or speed claim follows from selecting it.
