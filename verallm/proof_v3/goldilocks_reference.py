"""Pure-Python Goldilocks radix-2/LDE reference for future GPU golden vectors.

This module is deliberately unregistered: it is not imported from
``verallm.proof_v3``, has no payload or transcript format, and is not used by a
miner, validator, native library, or CUDA path.  It provides only canonical
Goldilocks field arithmetic, small radix-2 cosets, and deterministic low-degree
extension (LDE) values that a future GPU backend can compare against.

Goldilocks base-field arithmetic and an LDE are not a FRI proof, a commitment,
or the final soundness layer of an execution proof.  A production backend still
needs a qualified polynomial-commitment/FRI protocol, transcript binding,
queries, folding checks, and a verifier integration.

The reference intentionally caps materialized domains at ``2**16``.  This is a
CPU golden-vector safety limit, not a protocol parameter and not a future GPU
limit.  Inputs are canonical field integers only; the module never silently
reduces caller-supplied values or accepts packed field vectors.
"""

from __future__ import annotations

from functools import lru_cache

import operator
from dataclasses import dataclass, field
from typing import Final

from verallm.proof_v3.errors import ProofV3Error


GOLDILOCKS_MODULUS: Final = (1 << 64) - (1 << 32) + 1
GOLDILOCKS_BYTES: Final = 8
GOLDILOCKS_TWO_ADICITY: Final = 32
GOLDILOCKS_PRIMITIVE_GENERATOR: Final = 7
MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE: Final = 1 << 16
# Metadata/validation ceiling for the native (GPU-proven, CPU-verified)
# tier: statements and opening proofs may reference domains this large;
# CPU-side tree MATERIALIZATION stays bounded by the reference cap.
# Largest native RS/Merkle domain. 2^30 admits 2^28-variable columns
# (blowup 4): 1M-key capture columns need 2^27 values -> 2^29 codewords.
# The chain coset derivation folds this cap into its seed AND digest
# marker, so changing it can never silently cross-verify.
MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE: Final = 1 << 30


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def canonical_goldilocks(value: object, name: str = "Goldilocks value") -> int:
    """Validate one canonical Goldilocks element without implicit reduction."""

    integer = _integer(value, name)
    if integer < 0 or integer >= GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} is outside the canonical Goldilocks field")
    return integer


def goldilocks_to_bytes(value: object) -> bytes:
    """Encode one canonical Goldilocks field element as eight LE bytes."""

    return canonical_goldilocks(value).to_bytes(GOLDILOCKS_BYTES, "little")


def goldilocks_from_bytes(value: object) -> int:
    """Decode exactly eight canonical little-endian Goldilocks bytes."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProofV3Error("Goldilocks encoding must be bytes")
    encoded = bytes(value)
    if len(encoded) != GOLDILOCKS_BYTES:
        raise ProofV3Error("Goldilocks encoding must be exactly eight bytes")
    return canonical_goldilocks(
        int.from_bytes(encoded, "little"),
        "decoded Goldilocks value",
    )


def goldilocks_add(left: object, right: object) -> int:
    return (
        canonical_goldilocks(left, "Goldilocks add left")
        + canonical_goldilocks(right, "Goldilocks add right")
    ) % GOLDILOCKS_MODULUS


def goldilocks_sub(left: object, right: object) -> int:
    return (
        canonical_goldilocks(left, "Goldilocks sub left")
        - canonical_goldilocks(right, "Goldilocks sub right")
    ) % GOLDILOCKS_MODULUS


def goldilocks_mul(left: object, right: object) -> int:
    return (
        canonical_goldilocks(left, "Goldilocks mul left")
        * canonical_goldilocks(right, "Goldilocks mul right")
    ) % GOLDILOCKS_MODULUS


@lru_cache(maxsize=1 << 16)
def _inv_pow(scalar: int) -> int:
    return pow(scalar, GOLDILOCKS_MODULUS - 2, GOLDILOCKS_MODULUS)


def goldilocks_inv(value: object) -> int:
    """Return the canonical multiplicative inverse, rejecting zero."""

    scalar = canonical_goldilocks(value, "Goldilocks inverse value")
    if scalar == 0:
        raise ProofV3Error("Goldilocks zero has no multiplicative inverse")
    return _inv_pow(scalar)


def _radix2_size(value: object, name: str, *, cpu_materialized: bool) -> int:
    size = _integer(value, name)
    maximum = 1 << GOLDILOCKS_TWO_ADICITY
    if size < 1 or size > maximum or size & (size - 1):
        raise ProofV3Error(
            f"{name} must be a power of two in 1..=2**{GOLDILOCKS_TWO_ADICITY}"
        )
    if cpu_materialized and size > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
        raise ProofV3Error(
            f"{name} exceeds the CPU reference cap "
            f"{MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE}"
        )
    return size


@lru_cache(maxsize=128)
def _principal_root(size: int) -> int:
    root = pow(
        GOLDILOCKS_PRIMITIVE_GENERATOR,
        (GOLDILOCKS_MODULUS - 1) // size,
        GOLDILOCKS_MODULUS,
    )
    if pow(root, size, GOLDILOCKS_MODULUS) != 1:
        raise ProofV3Error("derived Goldilocks root does not have the requested order")
    if size > 1 and pow(root, size // 2, GOLDILOCKS_MODULUS) == 1:
        raise ProofV3Error("derived Goldilocks root is not principal")
    return root


def goldilocks_principal_root_of_unity(size: object) -> int:
    """Derive and validate the exact principal radix-2 root for ``size``.

    This primitive does not materialize a domain, so it supports every
    Goldilocks two-adic size through ``2**32``.  Domain and LDE functions apply
    the separate CPU materialization cap.
    """

    return _principal_root(_radix2_size(size, "Goldilocks root size", cpu_materialized=False))


@dataclass(frozen=True, slots=True)
class GoldilocksRadix2DomainReference:
    """A canonical, ordered multiplicative coset ``shift * <generator>``."""

    size: int
    shift: int = 1
    generator: int = field(init=False)

    def __post_init__(self) -> None:
        size = _radix2_size(
            self.size,
            "Goldilocks domain size",
            cpu_materialized=True,
        )
        shift = canonical_goldilocks(self.shift, "Goldilocks domain shift")
        if shift == 0:
            raise ProofV3Error("Goldilocks domain shift must be nonzero")
        root = _principal_root(size)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "generator", root)

    def points(self) -> tuple[int, ...]:
        """Return points in canonical FFT order: ``shift * generator**i``."""

        current = self.shift
        result: list[int] = []
        for _ in range(self.size):
            result.append(current)
            current = current * self.generator % GOLDILOCKS_MODULUS
        if current != self.shift:
            raise ProofV3Error("Goldilocks domain generator failed its order check")
        return tuple(result)


def goldilocks_radix2_domain_reference(
    *,
    size: object,
    shift: object = 1,
) -> GoldilocksRadix2DomainReference:
    """Construct a small canonical Goldilocks radix-2 coset domain."""

    return GoldilocksRadix2DomainReference(size=size, shift=shift)


def _field_vector(
    values: object,
    *,
    length: int | None,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable of canonical field integers")
    try:
        vector = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(
            f"{name} must be an iterable of canonical field integers"
        ) from exc
    if not vector:
        raise ProofV3Error(f"{name} must not be empty")
    if length is not None and len(vector) != length:
        raise ProofV3Error(f"{name} must contain exactly {length} values")
    if len(vector) > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
        raise ProofV3Error(f"{name} exceeds the CPU reference cap")
    return tuple(
        canonical_goldilocks(item, f"{name}[{index}]")
        for index, item in enumerate(vector)
    )


def _bit_reverse_permute(values: tuple[int, ...]) -> list[int]:
    result = list(values)
    index = 0
    for current in range(1, len(result)):
        bit = len(result) >> 1
        while index & bit:
            index ^= bit
            bit >>= 1
        index ^= bit
        if current < index:
            result[current], result[index] = result[index], result[current]
    return result


def _radix2_transform(values: tuple[int, ...], root: int) -> tuple[int, ...]:
    """Return ``sum_j values[j] * root**(i*j)`` using radix-2 butterflies."""

    result = _bit_reverse_permute(values)
    span = 2
    while span <= len(result):
        half = span // 2
        step = pow(root, len(result) // span, GOLDILOCKS_MODULUS)
        for start in range(0, len(result), span):
            twiddle = 1
            for offset in range(half):
                left = result[start + offset]
                right = result[start + offset + half] * twiddle % GOLDILOCKS_MODULUS
                result[start + offset] = (left + right) % GOLDILOCKS_MODULUS
                result[start + offset + half] = (left - right) % GOLDILOCKS_MODULUS
                twiddle = twiddle * step % GOLDILOCKS_MODULUS
        span <<= 1
    return tuple(result)


def ntt_goldilocks_reference(
    coefficients: object,
    *,
    domain: GoldilocksRadix2DomainReference,
) -> tuple[int, ...]:
    """Evaluate exactly ``domain.size`` coefficients on ``domain``.

    Coefficient ``j`` is multiplied by ``domain.shift**j`` before the radix-2
    transform, so the resulting vector is ordered by :meth:`domain.points`.
    """

    if not isinstance(domain, GoldilocksRadix2DomainReference):
        raise ProofV3Error("Goldilocks NTT domain has an unexpected type")
    vector = _field_vector(
        coefficients,
        length=domain.size,
        name="Goldilocks NTT coefficients",
    )
    shift_power = 1
    shifted: list[int] = []
    for coefficient in vector:
        shifted.append(coefficient * shift_power % GOLDILOCKS_MODULUS)
        shift_power = shift_power * domain.shift % GOLDILOCKS_MODULUS
    return _radix2_transform(tuple(shifted), domain.generator)


def intt_goldilocks_reference(
    evaluations: object,
    *,
    domain: GoldilocksRadix2DomainReference,
) -> tuple[int, ...]:
    """Interpolate coefficients from values in canonical ``domain.points`` order."""

    if not isinstance(domain, GoldilocksRadix2DomainReference):
        raise ProofV3Error("Goldilocks inverse NTT domain has an unexpected type")
    vector = _field_vector(
        evaluations,
        length=domain.size,
        name="Goldilocks inverse NTT evaluations",
    )
    inverse_root = goldilocks_inv(domain.generator)
    transformed = _radix2_transform(vector, inverse_root)
    inverse_size = goldilocks_inv(domain.size)
    inverse_shift = goldilocks_inv(domain.shift)
    shift_power = 1
    coefficients: list[int] = []
    for value in transformed:
        coefficients.append(
            value
            * inverse_size
            % GOLDILOCKS_MODULUS
            * shift_power
            % GOLDILOCKS_MODULUS
        )
        shift_power = shift_power * inverse_shift % GOLDILOCKS_MODULUS
    return tuple(coefficients)


def lde_goldilocks_reference(
    source_evaluations: object,
    *,
    target_size: object,
    source_shift: object = 1,
    target_shift: object = 1,
) -> tuple[int, ...]:
    """Extend source-coset evaluations to a larger target-coset domain.

    The source vector defines a polynomial of degree strictly below its radix-2
    source domain size.  It is interpolated exactly, zero-padded to the target
    size, and evaluated on the target coset.  Both domains are materialized by
    this CPU reference and therefore must remain within its explicit cap.
    """

    source = _field_vector(
        source_evaluations,
        length=None,
        name="Goldilocks LDE source evaluations",
    )
    source_size = _radix2_size(
        len(source),
        "Goldilocks LDE source_size",
        cpu_materialized=True,
    )
    target = _radix2_size(
        target_size,
        "Goldilocks LDE target_size",
        cpu_materialized=True,
    )
    if target < source_size or target % source_size:
        raise ProofV3Error(
            "Goldilocks LDE target_size must be a multiple of source_size"
        )
    source_domain = goldilocks_radix2_domain_reference(
        size=source_size,
        shift=source_shift,
    )
    target_domain = goldilocks_radix2_domain_reference(
        size=target,
        shift=target_shift,
    )
    coefficients = intt_goldilocks_reference(source, domain=source_domain)
    padded = coefficients + (0,) * (target - source_size)
    return ntt_goldilocks_reference(padded, domain=target_domain)


__all__ = [
    "GOLDILOCKS_BYTES",
    "GOLDILOCKS_MODULUS",
    "GOLDILOCKS_PRIMITIVE_GENERATOR",
    "GOLDILOCKS_TWO_ADICITY",
    "MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE",
    "GoldilocksRadix2DomainReference",
    "canonical_goldilocks",
    "goldilocks_add",
    "goldilocks_from_bytes",
    "goldilocks_inv",
    "goldilocks_mul",
    "goldilocks_principal_root_of_unity",
    "goldilocks_radix2_domain_reference",
    "goldilocks_sub",
    "goldilocks_to_bytes",
    "intt_goldilocks_reference",
    "lde_goldilocks_reference",
    "ntt_goldilocks_reference",
]
