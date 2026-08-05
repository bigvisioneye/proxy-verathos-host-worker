# Inference verification

Verathos uses Gleipnir proof protocol v3 to audit registered-model inference
without requiring validators to host the model weights or repeat generation.

The short version is:

1. every v3 response carries a canonical nonce-free light proof;
2. the miner cannot know in advance which indistinguishable canary will require
   a hard execution proof;
3. a hard proof opens nonce-selected parts of the committed runtime trace and
   checks them against authenticated model artifacts;
4. repeated audits, immediate probation and the complementary hot-capacity
   audit make cheaper substituted execution economically unattractive.

See [Gleipnir proof protocol v3](proof_protocol.md) for the wire and trust
details.

## Ordinary API requests

Organic requests use the nonce-free light tier. Tokens are streamed as the
model generates them. At the end of the stream, the miner supplies a canonical
commitment envelope, and the validator checks that it matches:

- the signed request and exact prompt tokens;
- the selected model, quantization and execution profile;
- the requested sampler parameters;
- the token IDs, text and finish reason observed by the validator;
- the miner's frozen runtime capture roots.

An accepted light response is useful evidence that the response and audit
state were frozen consistently. It is not a claim that every model operation
was opened and proven on that request. API and web clients therefore report
**Light proof accepted** for this tier.

## Unpredictable hard audits

The designated hard auditor also sends normal-looking canary requests. Before
the request, it commits a fresh secret nonce. Only after the miner freezes its
response envelope does the nonce determine whether hard verification is
required and which trace coordinates must be opened.

For a selected hard audit, the validator checks authenticated model weights,
selected projection equations, connected execution corridors, the applicable
attention or GDN transitions, and the terminal hidden-state/LM-head/sampler
path to the returned token. The exact relation set is fixed by the signed model
profile.

The audit is probabilistic. It proves the selected relations, not every
operation of every request. Canary frequency, selected-layer coverage and
failure policy are signed protocol parameters rather than miner-controlled
inputs.

## Streaming behavior

The final text token is never withheld for proof generation. Light-proof
work is completed at the end of ordinary inference. Hard proof construction
begins only after a hard nonce is revealed.

If a required hard proof is missing, malformed, late or invalid, the validator
records a proof failure and applies the configured probation/scoring policy.
Operational maintenance grace may temporarily suppress consequences, but it
does not change the verdict.

## Validator requirements

A validator needs:

- current chain and registry state;
- the authority-signed release index and model execution profile;
- the small authenticated calibration/commitment artifacts referenced by the
  profile;
- tokenizer metadata used for exact request binding;
- miner commitments and, for hard audits, the proof payload.

It does not need the model checkpoint or an inference GPU. Independent
validators can replay retained hard bundles; follower validators can instead
authenticate the designated auditor's signed exact-epoch verdict snapshot.

## Model support

The general v3 transcript is model- and quantization-independent, while each
runtime architecture is represented by a signed adapter profile. A new model
or quantization is admitted only after its profile is qualified and published
through the authenticated artifact index. Unknown combinations fail closed.

The signed profile also defines hard-audit context and decode reach. Light
serving may support larger contexts where policy permits, but a miner cannot
claim hard-audit coverage that its authenticated profile does not provide.

## Protocol rollout

Proof compatibility and maintenance forgiveness are separate:

- `[1,3]` allows updated validators to serve legacy v1 and v3 miners during the
  migration;
- `[3]` requires v3 after the update window;
- maintenance grace controls consequences for operational failures without
  changing which protocols are admitted.

Ordinary validators default to follower mode after updating. The configured
hard-auditor validator must explicitly run in local verification mode, and an
independent validator may opt into retained-bundle verification.

## What the result means

| Status | Interpretation |
|---|---|
| Light proof accepted | The v3 request, sampler, observed output and frozen commitment envelope match |
| Hard proof verified | The selected execution and terminal relations passed the signed profile |
| Verification failed | A required protocol object or relation was invalid |
| TEE verified | A separate hardware-attestation path was used instead of Gleipnir |

The protocol should not be summarized as deterministic proof of every operation
on every user request. Its production claim is probabilistic execution
integrity backed by unpredictable hard audits and capacity enforcement.
