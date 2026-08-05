# Economic model

Verathos is a Bittensor subnet for open-model inference. Miners provide model
serving, validators measure useful work and set weights, and gateways sell
OpenAI-compatible access to the network.

Gleipnir v3 is an enforcement mechanism, not a claim that every user request
contains a complete deterministic proof. Ordinary responses use light
commitments; unpredictable hard canaries audit registered execution; the
hot-capacity protocol checks retained compute and workspace.

## Why probabilistic verification

Repeating a large-model inference on every validator would consume the same
scarce GPU resources the network is trying to sell. Gleipnir instead separates
cheap always-on binding from unpredictable hard verification:

- miners cannot know before commitment which eligible canary will receive a
  hard challenge;
- selected hard relations are checked against authenticated model artifacts;
- invalid required proofs enter the immediate probation/scoring path;
- repeated coverage raises the expected cost of serving substituted work;
- capacity audits make it harder to advertise resources that are not actually
  retained.

The intended equilibrium is that serving the registered model honestly is
cheaper and more reliable than maintaining an alternate serving path plus the
state and compute needed to survive hard and capacity audits.

## Roles

### Miners

Miners register one or more model endpoints and earn weight from accepted
service. Their score reflects useful model capacity, measured request volume,
throughput, latency, demand and verification history.

Miner responsibilities include:

- serving the registered model and quantization;
- maintaining authenticated v3 capture state on every eligible response;
- producing a hard proof when the designated auditor reveals a valid nonce;
- participating in chain-timed hot-capacity audits;
- retaining validator-signed receipts for epoch accounting.

### Validators

Validators discover miners, run light canaries, score accepted work and set
Bittensor weights. The subnet-configured hard auditor also schedules the single
hard canary per endpoint. Other validators either authenticate its signed
exact-epoch verdict or explicitly replay retained hard bundles.

Follower mode reduces duplicated hard-proof traffic without allowing a missing
owner feed to become a miner failure: unavailable or invalid follower data is
neutral.

### Gateways

A gateway routes user traffic to eligible miners, handles authentication and
payments, and enforces the active proof-protocol allowlist. A successful v3
organic response exposes light-proof status. It does not relabel that
response as a hard proof.

## Epoch scoring

An epoch is 360 blocks (approximately 72 minutes at 12-second blocks).
Validators combine accepted organic and canary receipts into endpoint scores,
smooth them with an exponential moving average, and submit miner weights.

The implementation considers:

- registered model utility and quantization;
- input/output work and accepted throughput;
- time to first token and decode speed relative to peers;
- demand for the served model;
- proof, probation and capacity gates.

Exact coefficients are runtime policy and may change without changing the
proof protocol. Validators use authenticated observations and signed receipts,
not miner-reported throughput claims, as scoring inputs.

## Canary load

The designated hard auditor's signed policy uses two light canaries and one
hard canary per endpoint per epoch. Other validators send light canaries. The
hard decision and coordinates are derived after the response commitment, while
request lengths and prompts are secret-seeded to avoid an obvious audit-only
traffic class.

This gives endpoints with no organic demand enough observations to be scored;
the security model does not assume that a validator also operates a busy public
gateway.

## Proof failure and probation

A missing, malformed, late or invalid required hard proof is a real proof
failure. Outside maintenance grace it enters the configured immediate
failure/probation path and prevents normal routing or scoring as defined by
runtime policy.

Probation is rehabilitative rather than permanent. An endpoint returns after
the configured number of clean epochs. Repeated or prolonged failure can lead
to stronger availability action.

Maintenance grace is only an operational rollout tool. It records real
verdicts while suppressing selected consequences; it never changes the accepted
proof versions and never turns a failure into a cryptographic pass.

## Capacity and endpoint economics

Hot-capacity auditing is complementary to inference verification. It checks
that an endpoint can satisfy a chain-timed GPU workload using its advertised
resources. This helps deter multiple advertised endpoints from relying on the
same unavailable capacity.

Capacity is evaluated per registered endpoint and combined with inference-proof
history. Neither signal is sufficient on its own:

- capacity without execution binding does not establish which model served a
  response;
- a selected inference relation without retained serving capacity does not
  establish sustainable endpoint capacity.

## Payments

The gateway supports OpenAI-compatible paid inference. Deployment-specific
payment options may include account credit, TAO-backed credit, USDC and x402
pay-per-request settlement. Current routes and prices are returned by the live
gateway APIs; documentation examples are not a price oracle.

User settlement follows the gateway's accepted completion. A terminal protocol
failure does not create a successful receipt or settled usage for that attempt.

## On-chain anchors

Bittensor and the subnet contracts provide:

- miner and validator identities;
- model and endpoint registration;
- authenticated model roots and authority ownership;
- validator weight submission and subnet emissions;
- deployment-specific payment and usage records.

Large proof artifacts remain off-chain. Signed, content-addressed indexes bind
them to the chain ID, netuid and registry address.

## Security and governance boundaries

The protocol's production claim is probabilistic economic integrity:

- a light proof binds the request and output to what may later be audited;
- a hard success proves the selected registered relations;
- repeated audits and capacity checks make sustained substituted service
  costly and failure-prone;
- the system does not claim a full transformer SNARK for every response.

Hard-audit rates, relation coverage and model qualifications are signed policy.
Changing them requires a new authenticated profile or canary policy rather than
trusting miner-selected values.

## Related documentation

- [Gleipnir proof protocol v3](proof_protocol.md)
- [Inference verification](inference_protocol.md)
- [Bittensor integration](bittensor_integration.md)
- [Miner and validator setup](setup.md)
- [API reference](api.md)
