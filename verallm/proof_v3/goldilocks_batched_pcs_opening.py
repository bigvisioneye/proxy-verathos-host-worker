"""Batched terminal PCS opening: lockstep sumchecks + ONE RLC'd FRI chain.

Today every committed column runs its own interleaved sumcheck/FRI
opening with per-column layer trees.  This module opens EVERY column of
one batch through a single fold chain:

* **Lockstep sumchecks.**  Each column keeps its own degree-2 round
  polynomials (individually checked by the verifier), but all round
  polynomials of one round are absorbed into ONE transcript and every
  column folds with the SAME challenge.  Columns of different sizes join
  when the round size matches theirs (larger columns start earlier).

* **RLC'd layer chain.**  With shared challenges and the canonical
  shift-chain cosets (``coset_profile == "chain"``), same-round layers
  of different columns live on the same Reed-Solomon code, so their
  random linear combination ``D = sum_i rho^i * cw_i`` folds as one
  codeword: ``D_{r+1} = fold(D_r) + (joining components)``.  Only the
  D-chain commits layer trees; components contribute one base-layer
  multiopen each.  Proximity gaps give soundness for the RLC over the
  shared code; each column's sumcheck stays individually sound.

* **Queries.**  Nonce-derived base positions check the D-chain fold
  relation per layer, the RLC relation at every join layer (components'
  opened base leaves enter with their rho power), and the terminal
  constant ``sum_i rho^i * final_i``.

The verifier is pure host arithmetic + Merkle path checks; the prover
needs the fused CUDA kernels (device fold + streamed trees).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearPcsStatementV3,
    _QUERY_DOMAIN,
    _derive_field,
    _field,
    _layer_query_indices,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)

GOLDILOCKS_BATCHED_OPENING_ABI_V3 = (
    "goldilocks.multilinear_pcs.batched_opening.v1"
)

_BATCH_TRANSCRIPT_DOMAIN = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS_BATCH/V1/TRANSCRIPT/SHA256"
)
_BATCH_LAYER_DOMAIN = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS_BATCH/V1/LAYER/SHA256"
)

_INV2 = (GOLDILOCKS_MODULUS + 1) // 2


@dataclass(frozen=True, slots=True)
class BatchedOpeningComponentV3:
    """Prover-side input: one committed column and its terminal claim.

    ``values_host``: pinned-host stash uploaded lazily at the
    component's JOIN round instead of riding device-resident from the
    claims phase (pre-join residency otherwise stacks across every
    component).  Exactly one of ``values_device`` / ``values_host``
    is set.
    """

    tag: str
    statement: GoldilocksMultilinearPcsStatementV3
    tree: object
    values_device: object
    point: tuple[int, ...]
    claimed_value: int
    values_host: object = None


@dataclass(frozen=True, slots=True)
class BatchedComponentStatementV3:
    """Verifier-side input: the public face of one component."""

    tag: str
    statement: GoldilocksMultilinearPcsStatementV3
    commitment: bytes
    point: tuple[int, ...]
    claimed_value: int


@dataclass(frozen=True, slots=True)
class GoldilocksBatchedComponentOpeningV3:
    """One column's wire share: sumcheck rounds + base-layer multiopen."""

    tag: str
    round_polynomials: tuple[tuple[int, int, int], ...]
    final_value: int
    base_opening: GoldilocksMerkleMultiOpeningReference


@dataclass(frozen=True, slots=True)
class GoldilocksBatchedOpeningProofV3:
    """The batch's wire form: per-column shares + ONE layer chain."""

    components: tuple[GoldilocksBatchedComponentOpeningV3, ...]
    layer_commitments: tuple[bytes, ...]
    layer_openings: tuple[GoldilocksMerkleMultiOpeningReference, ...]


def _sorted_order(entries, statement_of, tag_of):
    order = sorted(
        entries,
        key=lambda e: (-statement_of(e).variable_count, tag_of(e)),
    )
    tags = [tag_of(e) for e in order]
    if len(set(tags)) != len(tags):
        raise ProofV3Error("batched opening component tags collide")
    return order


def _require_chain(statement) -> None:
    if statement.coset_profile != "chain":
        raise ProofV3Error(
            "batched opening requires the chain coset profile")


def _batch_seed(order, statement_of, commitment_of, point_of, claimed_of,
                validator_nonce: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_BATCH_TRANSCRIPT_DOMAIN)
    hasher.update(struct.pack("<I", len(order)))
    for entry in order:
        statement = statement_of(entry)
        point = point_of(entry)
        hasher.update(statement.digest())
        hasher.update(_fixed32(commitment_of(entry), "batch commitment"))
        hasher.update(struct.pack("<I", len(point)))
        for value in point:
            hasher.update(
                (_field(value, "batch point")).to_bytes(8, "little"))
        hasher.update(
            _field(claimed_of(entry), "batch claim").to_bytes(8, "little"))
    hasher.update(_fixed32(validator_nonce, "validator_nonce"))
    return hasher.digest()


def _batch_layer_binding(seed: bytes, layer_index: int) -> bytes:
    return hashlib.sha256(
        _BATCH_LAYER_DOMAIN + seed + struct.pack("<I", layer_index)
    ).digest()


def _query_positions(
    query_seed: bytes, half_base: int, query_count: int
) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                _QUERY_DOMAIN + query_seed + struct.pack("<I", index)
            ).digest()[:8],
            "little",
        )
        % half_base
        for index in range(query_count)
    ]


def _deg2_at(evals: tuple[int, int, int], z: int) -> int:
    g0, g1, g2 = evals
    return (
        g0 * ((z - 1) * (z - 2) % GOLDILOCKS_MODULUS)
        % GOLDILOCKS_MODULUS
        * _INV2
        - g1 * (z * (z - 2) % GOLDILOCKS_MODULUS)
        + g2 * (z * (z - 1) % GOLDILOCKS_MODULUS)
        % GOLDILOCKS_MODULUS
        * _INV2
    ) % GOLDILOCKS_MODULUS


def _device_eq_weights(fold_extension, point):
    """eq table over the hypercube on device, LSB variable first."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t
    from verallm.proof_v3.native_pcs_backend import _scalar_tensor

    weights = torch.ones(1, dtype=torch.int64, device="cuda")
    for r in point:
        one_minus = (1 - r) % GOLDILOCKS_MODULUS
        low = gl_mul_t(weights, _scalar_tensor(one_minus, weights))
        high = gl_mul_t(weights, _scalar_tensor(r % GOLDILOCKS_MODULUS,
                                                weights))
        weights = torch.cat([low, high])
    return weights


def _open_device_tree(tree, indices):
    if not hasattr(tree, "open_prepare"):
        # host-committed reference tree (small columns)
        return tree.open(indices)
    prepared = tree.open_prepare(indices)
    return tree.open_finish(
        prepared, prepared[3].cpu(), prepared[4].cpu())


# lockstep sumcheck states at or above this size live on PINNED HOST
# between rounds (streamed per chunk); below it they promote to device.
# Big components' v/w pairs stacking device-resident dominate the
# long-context open peak.  Overridable for deployments whose host
# memory limit is tighter than their spare VRAM.
import os as _os_vw

_VW_HOST_MIN_ELEMS = int(_os_vw.environ.get(
    "VERATHOS_PROOF_V3_VW_HOST_MIN", str(1 << 24)))


def _offload_state_pair(state) -> None:
    import torch

    for key in ("v", "w"):
        dev = state[key]
        host = torch.empty(dev.shape, dtype=dev.dtype, pin_memory=True)
        host.copy_(dev)
        state[key] = host
    state["host"] = True


def _promote_state_pair(state) -> None:
    state["v"] = state["v"].to("cuda", non_blocking=True)
    state["w"] = state["w"].to("cuda", non_blocking=True)
    state["host"] = False


def _upload_pair_chunk(scratch, src, half, start, stop):
    """(lo|hi) pair chunk of a pinned host state into device scratch.

    Two direct pinned H2D copies -- no host-side cat (which allocates a
    PAGEABLE buffer and forfeits the pinned fast path)."""

    length = stop - start
    view = scratch[: 2 * length]
    view[:length].copy_(src[start:stop], non_blocking=True)
    view[length:].copy_(
        src[half + start: half + stop], non_blocking=True)
    return view


def _host_state_partials(fold_extension, state):
    """round_partials over a host-resident (v, w), chunk-streamed."""

    import torch

    v, w = state["v"], state["w"]
    half = v.numel() // 2
    chunk = min(half, _STREAM_CHUNK)
    x_s = torch.empty(2 * chunk, dtype=v.dtype, device="cuda")
    f_s = torch.empty(2 * chunk, dtype=w.dtype, device="cuda")
    blocks = []
    for start in range(0, half, chunk):
        stop = min(start + chunk, half)
        x_c = _upload_pair_chunk(x_s, v, half, start, stop)
        f_c = _upload_pair_chunk(f_s, w, half, start, stop)
        blocks.append(fold_extension.round_partials(x_c, f_c))
    return torch.cat(blocks) if len(blocks) > 1 else blocks[0]


def _host_state_fold(fold_extension, state, encoded_challenge) -> None:
    """lerp_fold both host states, chunk-streamed; promote when small."""

    import torch

    scratch = None
    for key in ("v", "w"):
        src = state[key]
        half = src.numel() // 2
        chunk = min(half, _STREAM_CHUNK)
        if scratch is None or scratch.numel() < 2 * chunk:
            scratch = torch.empty(
                2 * chunk, dtype=src.dtype, device="cuda")
        dst = torch.empty(half, dtype=src.dtype, pin_memory=True)
        for start in range(0, half, chunk):
            stop = min(start + chunk, half)
            x_c = _upload_pair_chunk(scratch, src, half, start, stop)
            folded = fold_extension.lerp_fold(x_c, encoded_challenge)
            dst[start:stop].copy_(folded, non_blocking=True)
            del folded
        state[key] = dst
    if state["v"].numel() < _VW_HOST_MIN_ELEMS:
        _promote_state_pair(state)


def prove_goldilocks_batched_opening_v3(
    *,
    components: tuple[BatchedOpeningComponentV3, ...],
    validator_nonce: bytes,
    fused,
) -> GoldilocksBatchedOpeningProofV3:
    from verallm.proof_v3.native_cuda_fold_backend import _sum_partials
    from verallm.proof_v3.native_pcs_backend import (
        _bitrev_indices,
        _build_layer_tree,
        _encode_challenge,
        _to_int,
    )

    if not components:
        raise ProofV3Error("batched opening needs at least one component")
    for component in components:
        if component.statement.coset_profile != "chain":
            raise ProofV3Error(
                "batched opening component "
                f"{component.tag!r} does not use the chain coset profile"
            )
        if len(component.point) != component.statement.variable_count:
            raise ProofV3Error(
                "batched opening point arity does not match the statement")
    order = _sorted_order(
        components, lambda c: c.statement, lambda c: c.tag)
    top = order[0].statement
    if any(
        component.statement.query_count != top.query_count
        for component in order
    ):
        raise ProofV3Error(
            "batched opening components must share one PCS query count")
    n_max = top.variable_count
    base_size = top.codeword_size
    seed = _batch_seed(
        order,
        lambda c: c.statement,
        lambda c: c.tree.commitment,
        lambda c: c.point,
        lambda c: c.claimed_value,
        validator_nonce,
    )
    rho = _derive_field(seed, b"batchrlc", 0)
    transcript = seed
    fold_extension, tree_extension = fused
    accumulated = None
    active: list[dict] = []
    layer_trees = []
    layer_roots: list[bytes] = []
    join_at = {}
    for index, component in enumerate(order):
        join_at.setdefault(
            n_max - component.statement.variable_count, []
        ).append((index, component))

    powers = {}
    acc = 1
    for index in range(len(order)):
        powers[index] = acc
        acc = acc * rho % GOLDILOCKS_MODULUS

    import os as _os
    import torch as _torch

    _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"

    def _mem(label: str) -> None:
        if _trace:
            print(
                f"BOPEN {label}: alloc="
                f"{_torch.cuda.memory_allocated() / 1e9:.2f} peak="
                f"{_torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                flush=True)

    _mem("start")
    for round_index in range(n_max):
        for index, component in join_at.get(round_index, ()):
            accumulated = _accumulate_component(
                fold_extension, accumulated, component, powers[index],
                mutable=round_index == 0)
            variables = component.statement.variable_count
            values_device = component.values_device
            if values_device is None:
                stash = component.values_host
                if stash is not None:
                    # lazy join upload: the stash waited on host instead
                    # of stacking device-resident across all pre-join
                    # components
                    values_device = stash.to("cuda", non_blocking=True)
                    object.__setattr__(component, "values_host", None)
            if values_device is None:
                raise ProofV3Error(
                    "batched opening component lacks device values")
            rev = _bitrev_indices(variables)
            state = {
                "component": component,
                "v": values_device.index_select(0, rev),
                "w": _device_eq_weights(
                    fold_extension, component.point
                ).index_select(0, rev),
                "rounds": [],
                "host": False,
            }
            # v/w carry the sumcheck from here; drop the last strong
            # reference to the column's original evaluation vector so
            # the join workspace does not stack across components
            object.__setattr__(component, "values_device", None)
            del values_device
            if state["v"].numel() >= _VW_HOST_MIN_ELEMS:
                # big states wait on pinned host between rounds instead
                # of stacking device-resident across components
                _offload_state_pair(state)
            active.append(state)
        blob = b""
        for state in active:
            if state["host"]:
                partials = _host_state_partials(fold_extension, state)
            else:
                partials = fold_extension.round_partials(
                    state["v"], state["w"])
            g0, g1, g2 = _sum_partials(partials)
            del partials
            state["rounds"].append((g0, g1, g2))
            blob += (
                g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            )
        transcript = hashlib.sha256(transcript + blob).digest()
        challenge = _derive_field(transcript, b"batchfold", round_index)
        encoded = _encode_challenge(challenge)
        for state in active:
            if state["host"]:
                _host_state_fold(fold_extension, state, encoded)
            else:
                state["v"] = fold_extension.lerp_fold(state["v"], encoded)
                state["w"] = fold_extension.lerp_fold(state["w"], encoded)
        if round_index < 3:
            _mem(f"round{round_index} pre-fold")
        accumulated = _fold_layer(
            fold_extension, accumulated, top, round_index, encoded)
        if round_index < 3:
            _mem(f"round{round_index} post-fold")
        folded_tree = _build_layer_tree(
            tree_extension, accumulated,
            binding_digest=_batch_layer_binding(seed, round_index + 1),
            lazy_commit=False)
        if round_index < 3:
            _mem(f"round{round_index} post-tree")
        layer_trees.append(folded_tree)
        layer_roots.append(folded_tree.commitment)
        transcript = hashlib.sha256(
            transcript + folded_tree.commitment).digest()

    _mem("pre-queries")
    query_seed = hashlib.sha256(transcript + b"queries").digest()
    positions = _query_positions(
        query_seed, base_size // 2, top.query_count
    )

    wire_components = []
    for state in active:
        component = state["component"]
        base_indices = _layer_query_indices(
            positions=positions,
            size=component.statement.codeword_size,
            is_final=False,
        )
        wire_components.append(GoldilocksBatchedComponentOpeningV3(
            tag=component.tag,
            round_polynomials=tuple(state["rounds"]),
            final_value=_to_int(state["v"].cpu()[0].item()),
            base_opening=_open_device_tree(component.tree, base_indices),
        ))
    layer_openings = []
    for layer_index, layer_tree in enumerate(layer_trees, start=1):
        layer_openings.append(_open_device_tree(
            layer_tree,
            _layer_query_indices(
                positions=positions,
                size=base_size >> layer_index,
                is_final=layer_index == n_max,
            ),
        ))
    return GoldilocksBatchedOpeningProofV3(
        components=tuple(wire_components),
        layer_commitments=tuple(layer_roots),
        layer_openings=tuple(layer_openings),
    )


# join/fold streaming chunk.  With the fused axpy/fold kernels the
# per-chunk transient is zero, so the chunk only bounds the H2D upload
# granularity; the torch-limb FALLBACK still allocates ~15x the chunk,
# which this size keeps under ~2GB.
_STREAM_CHUNK = 1 << 23


def _enc_scalar(value: int) -> int:
    """Canonical field element -> int64 two's-complement bit pattern."""

    value %= GOLDILOCKS_MODULUS
    return value - (1 << 64) if value >= (1 << 63) else value


def _component_base_codeword(fold_extension, component):
    """Base codeword of one component (committed leaves, else re-encode)."""

    import torch

    from verallm.proof_v3.goldilocks_reference import (
        goldilocks_principal_root_of_unity,
    )
    from verallm.proof_v3.native_cuda_fold_backend import (
        fused_ntt_goldilocks,
    )

    statement = component.statement
    size = statement.codeword_size
    stored = getattr(component.tree, "_values", None)
    if (
        isinstance(stored, torch.Tensor)
        and stored.numel() == size
        and stored.dtype == torch.int64
    ):
        return stored
    values = component.values_device
    if values is None:
        stash = component.values_host
        if stash is not None:
            # rare fallback (tree without stored leaves): resume the
            # stash once and pin it for the join's v/w init
            values = stash.to("cuda", non_blocking=True)
            object.__setattr__(component, "values_device", values)
            object.__setattr__(component, "values_host", None)
    if values is None:
        raise ProofV3Error(
            "batched opening component lacks both committed leaves "
            "and device values")
    padded = torch.zeros(size, dtype=torch.int64, device="cuda")
    padded[: values.numel()] = values
    return fused_ntt_goldilocks(
        fold_extension, padded,
        shift=statement.domain_shift(0),
        generator=goldilocks_principal_root_of_unity(size),
        mutable=True)


def _accumulate_component(fold_extension, accumulated, component,
                          rho_power: int, *, mutable: bool):
    """target = accumulated + rho^k * codeword, chunk-streamed.

    The committed leaves upload PER CHUNK (the full component codeword
    never materializes on device beside the accumulator), so transients
    stay bounded by the chunk size at any context length.  Once a layer
    tree references the accumulator its buffer is FROZEN (the tree
    opens against those exact leaves later): joins after round 0 write
    a fresh buffer instead of mutating in place.
    """

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_scale_t,
    )

    source = _component_base_codeword(fold_extension, component)
    size = source.numel()
    fresh = accumulated is None
    if fresh:
        target = torch.empty(size, dtype=torch.int64, device="cuda")
    elif accumulated.numel() != size:
        raise ProofV3Error(
            "batched opening join size does not match the accumulator")
    elif mutable:
        target = accumulated
    else:
        target = torch.empty_like(accumulated)
    fused_axpy = hasattr(fold_extension, "gl_axpy_out")
    for start in range(0, size, _STREAM_CHUNK):
        stop = min(start + _STREAM_CHUNK, size)
        piece = source[start:stop]
        piece = piece.cuda() if not piece.is_cuda else piece.contiguous()
        if fresh:
            if rho_power != 1:
                piece = gl_scale_t(piece, rho_power)
            target[start:stop].copy_(piece)
        elif fused_axpy:
            # single fused kernel, zero intermediates (safe in place:
            # element i reads acc[i] and writes out[i] only)
            fold_extension.gl_axpy_out(
                accumulated[start:stop], piece, _enc_scalar(rho_power),
                target[start:stop])
        else:
            if rho_power != 1:
                piece = gl_scale_t(piece, rho_power)
            target[start:stop].copy_(
                gl_add_t(accumulated[start:stop], piece))
        del piece
    return target


def _fold_layer(fold_extension, codeword, statement, round_index: int,
                encoded_challenge: int):
    """One FRI fold on the codeword, chunk-streamed (byte-identical).

    Per element: even = lo + hi, odd = (lo - hi) / (s_j * g^i), folded
    = ((1-c) * even + c * odd) / 2 -- exactly the verifier equation,
    evaluated per chunk with a bounded twiddle table (g^-i factors as
    g^-start * table) and the half-split lerp kernel.
    """

    import torch

    from verallm.proof_v3.goldilocks_reference import (
        GOLDILOCKS_MODULUS as _P,
        goldilocks_inv,
        goldilocks_principal_root_of_unity,
    )
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_mul_t,
        gl_scale_t,
        gl_sub_t,
    )
    from verallm.proof_v3.native_pcs_backend import _gl_scalar_powers

    size = codeword.numel()
    half = size // 2
    inverse_shift = goldilocks_inv(statement.domain_shift(round_index))
    inverse_generator = goldilocks_inv(
        goldilocks_principal_root_of_unity(size))
    folded = torch.empty(half, dtype=torch.int64, device="cuda")
    chunk = min(half, _STREAM_CHUNK)
    base_powers = _gl_scalar_powers(inverse_generator, chunk)
    fused_step = hasattr(fold_extension, "gl_fold_step_out")
    for start in range(0, half, chunk):
        stop = min(start + chunk, half)
        scale = (
            inverse_shift * pow(inverse_generator, start, _P)
        ) % _P
        if fused_step:
            # the whole per-element fold in ONE kernel writing straight
            # into the output slice -- no intermediates at all
            fold_extension.gl_fold_step_out(
                codeword[start:stop],
                codeword[half + start: half + stop],
                base_powers[: stop - start], _enc_scalar(scale),
                encoded_challenge, _enc_scalar(_INV2),
                folded[start:stop])
            continue
        low = codeword[start:stop].contiguous()
        high = codeword[half + start: half + stop].contiguous()
        total = gl_add_t(low, high)
        twiddles = gl_scale_t(base_powers[: stop - start], scale)
        odd = gl_mul_t(gl_sub_t(low, high), twiddles)
        del low, high, twiddles
        paired = torch.cat((total, odd))
        del total, odd
        piece = fold_extension.lerp_fold(paired, encoded_challenge)
        del paired
        folded[start:stop].copy_(gl_scale_t(piece, _INV2))
        del piece
    return folded


def verify_goldilocks_batched_opening_v3(
    proof: object,
    *,
    components: tuple[BatchedComponentStatementV3, ...],
    validator_nonce: bytes,
) -> None:
    try:
        if not isinstance(proof, GoldilocksBatchedOpeningProofV3):
            raise ProofV3VerificationError(
                "batched opening proof type is wrong")
        if not components:
            raise ProofV3VerificationError(
                "batched opening needs at least one component")
        for component in components:
            _require_chain(component.statement)
            if len(component.point) != component.statement.variable_count:
                raise ProofV3VerificationError(
                    "batched opening point arity is wrong")
        order = _sorted_order(
            components, lambda c: c.statement, lambda c: c.tag)
        if tuple(c.tag for c in order) != tuple(
            c.tag for c in proof.components
        ):
            raise ProofV3VerificationError(
                "batched opening component order is wrong")
        top = order[0].statement
        if any(
            component.statement.query_count != top.query_count
            for component in order
        ):
            raise ProofV3VerificationError(
                "batched opening components use mixed PCS query counts")
        n_max = top.variable_count
        base_size = top.codeword_size
        if len(proof.layer_commitments) != n_max or len(
            proof.layer_openings
        ) != n_max:
            raise ProofV3VerificationError(
                "batched opening layer count is wrong")
        seed = _batch_seed(
            order,
            lambda c: c.statement,
            lambda c: c.commitment,
            lambda c: c.point,
            lambda c: c.claimed_value,
            validator_nonce,
        )
        rho = _derive_field(seed, b"batchrlc", 0)
        powers = {}
        acc = 1
        for index in range(len(order)):
            powers[index] = acc
            acc = acc * rho % GOLDILOCKS_MODULUS

        join_of = {
            index: n_max - component.statement.variable_count
            for index, component in enumerate(order)
        }
        runnings = {
            index: _field(component.claimed_value, "batch claim")
            for index, component in enumerate(order)
        }
        local_round = {index: 0 for index in range(len(order))}
        transcript = seed
        challenges: list[int] = []
        for round_index in range(n_max):
            blob = b""
            for index, component in enumerate(order):
                if join_of[index] > round_index:
                    continue
                rounds = proof.components[index].round_polynomials
                j = local_round[index]
                if j >= len(rounds):
                    raise ProofV3VerificationError(
                        "batched opening round arity is wrong")
                g0, g1, g2 = (
                    _field(rounds[j][0], "g0"),
                    _field(rounds[j][1], "g1"),
                    _field(rounds[j][2], "g2"),
                )
                if (g0 + g1) % GOLDILOCKS_MODULUS != runnings[index]:
                    raise ProofV3VerificationError(
                        "batched opening round sum is wrong")
                blob += (
                    g0.to_bytes(8, "little")
                    + g1.to_bytes(8, "little")
                    + g2.to_bytes(8, "little")
                )
            transcript = hashlib.sha256(transcript + blob).digest()
            challenge = _derive_field(transcript, b"batchfold", round_index)
            challenges.append(challenge)
            for index, component in enumerate(order):
                if join_of[index] > round_index:
                    continue
                j = local_round[index]
                rounds = proof.components[index].round_polynomials
                runnings[index] = _deg2_at(
                    tuple(
                        _field(v, "round value") for v in rounds[j]
                    ),
                    challenge,
                )
                local_round[index] = j + 1
            transcript = hashlib.sha256(
                transcript + _fixed32(
                    proof.layer_commitments[round_index],
                    "batch layer commitment",
                )
            ).digest()
        for index, component in enumerate(order):
            wire = proof.components[index]
            variables = component.statement.variable_count
            if len(wire.round_polynomials) != variables:
                raise ProofV3VerificationError(
                    "batched opening round count is wrong")
            eq_product = 1
            for j in range(variables):
                c = challenges[join_of[index] + j]
                r = _field(component.point[j], "batch point")
                eq_product = (
                    eq_product * (((1 - c) * (1 - r) + c * r)
                                  % GOLDILOCKS_MODULUS)
                ) % GOLDILOCKS_MODULUS
            final = _field(wire.final_value, "final value")
            if runnings[index] != final * eq_product % GOLDILOCKS_MODULUS:
                raise ProofV3VerificationError(
                    "batched opening final coupling fails")

        query_seed = hashlib.sha256(transcript + b"queries").digest()
        positions = _query_positions(
            query_seed, base_size // 2, top.query_count
        )

        # Merkle-verify every opening and index them.
        component_values: list[dict[int, int]] = []
        for index, component in enumerate(order):
            wire = proof.components[index]
            size = component.statement.codeword_size
            expected_indices = _layer_query_indices(
                positions=positions, size=size, is_final=False)
            verify_goldilocks_merkle_multiopening_reference(
                component.commitment,
                wire.base_opening,
                expected_binding_digest=(
                    component.statement.layer_binding_digest(0)),
                expected_leaf_count=size,
                expected_leaf_width=1,
                expected_indices=expected_indices,
            )
            component_values.append({
                opened_index: row[0]
                for opened_index, row in zip(
                    wire.base_opening.indices,
                    wire.base_opening.rows,
                    strict=True,
                )
            })
        layer_values: list[dict[int, int]] = []
        for layer_index, opening in enumerate(
            proof.layer_openings, start=1
        ):
            size = base_size >> layer_index
            verify_goldilocks_merkle_multiopening_reference(
                proof.layer_commitments[layer_index - 1],
                opening,
                expected_binding_digest=_batch_layer_binding(
                    seed, layer_index),
                expected_leaf_count=size,
                expected_leaf_width=1,
                expected_indices=_layer_query_indices(
                    positions=positions, size=size,
                    is_final=layer_index == n_max,
                ),
            )
            layer_values.append({
                opened_index: row[0]
                for opened_index, row in zip(
                    opening.indices, opening.rows, strict=True)
            })

        def _rlc_at(layer_index: int, position_index: int) -> int:
            total = 0
            for index, component in enumerate(order):
                if join_of[index] != layer_index:
                    continue
                total = (
                    total
                    + powers[index]
                    * component_values[index][position_index]
                ) % GOLDILOCKS_MODULUS
            return total

        for round_index in range(n_max):
            size = base_size >> round_index
            half = size // 2
            challenge = challenges[round_index]
            for position in positions:
                folded_position = position % half
                if round_index == 0:
                    value_pos = _rlc_at(0, folded_position)
                    value_neg = _rlc_at(0, folded_position + half)
                else:
                    value_pos = layer_values[round_index - 1][
                        folded_position]
                    value_neg = layer_values[round_index - 1][
                        folded_position + half]
                    value_pos = (
                        value_pos + _rlc_at(round_index, folded_position)
                    ) % GOLDILOCKS_MODULUS
                    value_neg = (
                        value_neg
                        + _rlc_at(round_index, folded_position + half)
                    ) % GOLDILOCKS_MODULUS
                x = top.domain_point(round_index, folded_position)
                even = (
                    (value_pos + value_neg) % GOLDILOCKS_MODULUS * _INV2
                ) % GOLDILOCKS_MODULUS
                odd = (
                    (value_pos - value_neg)
                    % GOLDILOCKS_MODULUS
                    * _INV2
                    % GOLDILOCKS_MODULUS
                    * goldilocks_inv(x)
                ) % GOLDILOCKS_MODULUS
                expected_fold = (
                    (1 - challenge) * even + challenge * odd
                ) % GOLDILOCKS_MODULUS
                # the committed child layer contains the joins already;
                # join terms enter on the PARENT side of the next round.
                target = position % half
                child = layer_values[round_index][target]
                if expected_fold != child:
                    raise ProofV3VerificationError(
                        "batched opening fold consistency fails")
        terminal = 0
        for index, component in enumerate(order):
            terminal = (
                terminal
                + powers[index]
                * _field(
                    proof.components[index].final_value, "final value")
            ) % GOLDILOCKS_MODULUS
        for value in layer_values[n_max - 1].values():
            if value != terminal:
                raise ProofV3VerificationError(
                    "batched opening terminal is not the RLC constant")
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "batched opening proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_BATCHED_OPENING_ABI_V3",
    "BatchedComponentStatementV3",
    "BatchedOpeningComponentV3",
    "GoldilocksBatchedComponentOpeningV3",
    "GoldilocksBatchedOpeningProofV3",
    "prove_goldilocks_batched_opening_v3",
    "verify_goldilocks_batched_opening_v3",
]
