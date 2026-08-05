"""Strict JSON document for signed proof-v2 static manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Collection

from verallm.proof_v2.manifest import (
    ExecutionAuditPolicyV2,
    LayerExecutionDescriptor,
    ManifestFormatError,
    ModelExecutionDescriptor,
    ModelSpecIdentity,
    OperationDescriptor,
    StaticWeightCommitmentManifest,
    verify_signed_manifest,
)

PCS_SUITE = "pallas-pedersen-ipa-v1"
PCS_GENERATOR_VERSION = "verathos-pcs-v2-pallas-pedersen-gens-v1"
MAX_MANIFEST_DOCUMENT_BYTES = 32 << 20
MAX_SIGNATURES = 256


class ManifestDocumentError(ValueError):
    """The signed manifest JSON document is malformed or noncanonical."""


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestDocumentError(
                f"manifest document contains duplicate field: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ManifestDocumentError(
        f"manifest document contains a non-finite number: {value}"
    )


def model_spec_identity_to_dict(spec: ModelSpecIdentity) -> dict:
    """Return the canonical JSON object for an exact on-chain ModelSpec copy."""

    if not isinstance(spec, ModelSpecIdentity):
        raise ManifestDocumentError("model_spec has an unexpected type")
    return {
        "model_id": spec.model_id,
        "weight_merkle_root": spec.weight_merkle_root.hex(),
        "layer_roots": [root.hex() for root in spec.layer_roots],
        "num_layers": spec.num_layers,
        "hidden_dim": spec.hidden_dim,
        "intermediate_dim": spec.intermediate_dim,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "vocab_size": spec.vocab_size,
        "quant_mode": spec.quant_mode,
        "merkle_chunk_size": spec.merkle_chunk_size,
        "activation": spec.activation,
        "norm_type": spec.norm_type,
        "attention_type": spec.attention_type,
        "num_experts": spec.num_experts,
        "expert_w_num_cols": spec.expert_w_num_cols,
        "lm_head_root": spec.lm_head_root.hex(),
        "embedding_root": spec.embedding_root.hex(),
        "weight_file_hash": spec.weight_file_hash.hex(),
        "tokenizer_hash": spec.tokenizer_hash.hex(),
    }


def model_spec_identity_from_dict(value: object) -> ModelSpecIdentity:
    """Strictly parse the canonical JSON object for an exact ModelSpec copy."""

    spec_names = {field.name for field in fields(ModelSpecIdentity)}
    spec_value = _object(value, spec_names, "model_spec")
    layer_roots_raw = _list(spec_value["layer_roots"], "layer_roots")
    try:
        return ModelSpecIdentity(
            model_id=_text(spec_value["model_id"], "model_id"),
            weight_merkle_root=_hex(
                spec_value["weight_merkle_root"], 32, "weight_merkle_root"
            ),
            layer_roots=tuple(
                _hex(item, 32, f"layer_roots[{index}]")
                for index, item in enumerate(layer_roots_raw)
            ),
            num_layers=_integer(spec_value["num_layers"], "num_layers"),
            hidden_dim=_integer(spec_value["hidden_dim"], "hidden_dim"),
            intermediate_dim=_integer(
                spec_value["intermediate_dim"], "intermediate_dim"
            ),
            num_heads=_integer(spec_value["num_heads"], "num_heads"),
            head_dim=_integer(spec_value["head_dim"], "head_dim"),
            vocab_size=_integer(spec_value["vocab_size"], "vocab_size"),
            quant_mode=_text(spec_value["quant_mode"], "quant_mode"),
            merkle_chunk_size=_integer(
                spec_value["merkle_chunk_size"], "merkle_chunk_size"
            ),
            activation=_text(spec_value["activation"], "activation"),
            norm_type=_text(spec_value["norm_type"], "norm_type"),
            attention_type=_text(spec_value["attention_type"], "attention_type"),
            num_experts=_integer(spec_value["num_experts"], "num_experts"),
            expert_w_num_cols=_integer(
                spec_value["expert_w_num_cols"], "expert_w_num_cols"
            ),
            lm_head_root=_hex(spec_value["lm_head_root"], 32, "lm_head_root"),
            embedding_root=_hex(spec_value["embedding_root"], 32, "embedding_root"),
            weight_file_hash=_hex(
                spec_value["weight_file_hash"], 32, "weight_file_hash"
            ),
            tokenizer_hash=_hex(spec_value["tokenizer_hash"], 32, "tokenizer_hash"),
        )
    except ManifestFormatError as exc:
        raise ManifestDocumentError("model_spec body is not canonical") from exc


def manifest_to_dict(manifest: StaticWeightCommitmentManifest) -> dict:
    """Return the canonical JSON object for an unsigned manifest body."""

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ManifestDocumentError("manifest has an unexpected type")
    return {
        "protocol_version": manifest.protocol_version,
        "chain_id": manifest.chain_id,
        "netuid": manifest.netuid,
        "registry_address": manifest.registry_address.hex(),
        "model_spec": model_spec_identity_to_dict(manifest.model_spec),
        "pcs_suite": manifest.pcs_suite,
        "pcs_generator_version": manifest.pcs_generator_version,
        "execution_profile": manifest.execution_profile,
        "layer_execution": [
            {
                "layer": descriptor.layer,
                "attention_profile": descriptor.attention_profile,
                "bridge_parameter_root": descriptor.bridge_parameter_root.hex(),
                "input_norm_weight_f16": descriptor.input_norm_weight_f16.hex(),
                "post_attention_norm_weight_f16": (
                    descriptor.post_attention_norm_weight_f16.hex()
                ),
                "norm_epsilon_q32": descriptor.norm_epsilon_q32,
                "transition_profile": descriptor.transition_profile,
                "transition_parameter_root": (
                    descriptor.transition_parameter_root.hex()
                ),
                "transition_parameters": descriptor.transition_parameters.hex(),
            }
            for descriptor in manifest.layer_execution
        ],
        "model_execution": (
            None
            if manifest.model_execution is None
            else {
                "embedding_scale_q32": (manifest.model_execution.embedding_scale_q32),
                "final_norm_weight_f16": (
                    manifest.model_execution.final_norm_weight_f16.hex()
                ),
                "final_norm_epsilon_q32": (
                    manifest.model_execution.final_norm_epsilon_q32
                ),
                "parameter_root": manifest.model_execution.parameter_root.hex(),
                **(
                    {
                        "audit_policy": {
                            "hard_audit_bps": (
                                manifest.model_execution.audit_policy.hard_audit_bps
                            ),
                            "hard_layer_count": (
                                manifest.model_execution.audit_policy.hard_layer_count
                            ),
                            "hard_blocks_per_operation": (
                                manifest.model_execution.audit_policy.hard_blocks_per_operation
                            ),
                            "min_full_attention_layers": (
                                manifest.model_execution.audit_policy.min_full_attention_layers
                            ),
                            "min_gdn_layers": (
                                manifest.model_execution.audit_policy.min_gdn_layers
                            ),
                            "full_attention_heads_per_layer": (
                                manifest.model_execution.audit_policy.full_attention_heads_per_layer
                            ),
                        }
                    }
                    if manifest.model_execution.audit_policy is not None
                    else {}
                ),
            }
        ),
        "operations": [
            {
                "layer": operation.layer,
                "operation_id": operation.operation_id,
                "expert_id": operation.expert_id,
                "rows": operation.rows,
                "cols": operation.cols,
                "commitment": operation.commitment.hex(),
                "weight_scale_q32": operation.weight_scale_q32,
                **(
                    {"weight_block_scales_q32": list(operation.weight_block_scales_q32)}
                    if operation.weight_block_scales_q32
                    else {}
                ),
                "min_x_scale_q32": operation.min_x_scale_q32,
                "runtime_abs_tolerance_q32": operation.runtime_abs_tolerance_q32,
                "runtime_rel_tolerance_bps": operation.runtime_rel_tolerance_bps,
            }
            for operation in manifest.operations
        ],
    }


def manifest_from_dict(value: object) -> StaticWeightCommitmentManifest:
    """Strictly parse a canonical unsigned manifest JSON object."""

    manifest_value = _object(
        value,
        {
            "protocol_version",
            "chain_id",
            "netuid",
            "registry_address",
            "model_spec",
            "pcs_suite",
            "pcs_generator_version",
            "execution_profile",
            "layer_execution",
            "model_execution",
            "operations",
        },
        "manifest",
    )
    spec = model_spec_identity_from_dict(manifest_value["model_spec"])
    operations_raw = _list(manifest_value["operations"], "operations")
    execution_profile = manifest_value["execution_profile"]
    if execution_profile is not None:
        execution_profile = _text(execution_profile, "execution_profile")
    layer_execution_raw = _list(
        manifest_value["layer_execution"],
        "layer_execution",
    )
    layer_execution = []
    for index, raw in enumerate(layer_execution_raw):
        descriptor = _object(
            raw,
            {
                "layer",
                "attention_profile",
                "bridge_parameter_root",
                "input_norm_weight_f16",
                "post_attention_norm_weight_f16",
                "norm_epsilon_q32",
                "transition_profile",
                "transition_parameter_root",
                "transition_parameters",
            },
            f"layer_execution[{index}]",
        )
        try:
            layer_execution.append(
                LayerExecutionDescriptor(
                    layer=_integer(
                        descriptor["layer"],
                        f"layer_execution[{index}].layer",
                    ),
                    attention_profile=_text(
                        descriptor["attention_profile"],
                        f"layer_execution[{index}].attention_profile",
                    ),
                    bridge_parameter_root=_hex(
                        descriptor["bridge_parameter_root"],
                        32,
                        f"layer_execution[{index}].bridge_parameter_root",
                    ),
                    input_norm_weight_f16=_variable_hex(
                        descriptor["input_norm_weight_f16"],
                        f"layer_execution[{index}].input_norm_weight_f16",
                        allow_empty=True,
                    ),
                    post_attention_norm_weight_f16=_variable_hex(
                        descriptor["post_attention_norm_weight_f16"],
                        f"layer_execution[{index}].post_attention_norm_weight_f16",
                        allow_empty=True,
                    ),
                    norm_epsilon_q32=_integer(
                        descriptor["norm_epsilon_q32"],
                        f"layer_execution[{index}].norm_epsilon_q32",
                    ),
                    transition_profile=(
                        None
                        if descriptor["transition_profile"] is None
                        else _text(
                            descriptor["transition_profile"],
                            f"layer_execution[{index}].transition_profile",
                        )
                    ),
                    transition_parameter_root=_variable_hex(
                        descriptor["transition_parameter_root"],
                        f"layer_execution[{index}].transition_parameter_root",
                        allow_empty=True,
                    ),
                    transition_parameters=_variable_hex(
                        descriptor["transition_parameters"],
                        f"layer_execution[{index}].transition_parameters",
                        allow_empty=True,
                    ),
                )
            )
        except ManifestFormatError as exc:
            raise ManifestDocumentError(
                f"layer_execution[{index}] body is not canonical"
            ) from exc
    model_execution_raw = manifest_value["model_execution"]
    model_execution = None
    if model_execution_raw is not None:
        descriptor = _object(
            model_execution_raw,
            {
                "embedding_scale_q32",
                "final_norm_weight_f16",
                "final_norm_epsilon_q32",
                "parameter_root",
            }
            | (
                {"audit_policy"}
                if isinstance(model_execution_raw, dict)
                and "audit_policy" in model_execution_raw
                else set()
            ),
            "model_execution",
        )
        try:
            audit_policy_raw = descriptor.get("audit_policy")
            audit_policy = None
            if audit_policy_raw is not None:
                audit_policy_value = _object(
                    audit_policy_raw,
                    {
                        "hard_audit_bps",
                        "hard_layer_count",
                        "hard_blocks_per_operation",
                        "min_full_attention_layers",
                        "min_gdn_layers",
                        "full_attention_heads_per_layer",
                    },
                    "model_execution.audit_policy",
                )
                audit_policy = ExecutionAuditPolicyV2(
                    hard_audit_bps=_integer(
                        audit_policy_value["hard_audit_bps"],
                        "model_execution.audit_policy.hard_audit_bps",
                    ),
                    hard_layer_count=_integer(
                        audit_policy_value["hard_layer_count"],
                        "model_execution.audit_policy.hard_layer_count",
                    ),
                    hard_blocks_per_operation=_integer(
                        audit_policy_value["hard_blocks_per_operation"],
                        "model_execution.audit_policy.hard_blocks_per_operation",
                    ),
                    min_full_attention_layers=_integer(
                        audit_policy_value["min_full_attention_layers"],
                        "model_execution.audit_policy.min_full_attention_layers",
                    ),
                    min_gdn_layers=_integer(
                        audit_policy_value["min_gdn_layers"],
                        "model_execution.audit_policy.min_gdn_layers",
                    ),
                    full_attention_heads_per_layer=_integer(
                        audit_policy_value["full_attention_heads_per_layer"],
                        "model_execution.audit_policy.full_attention_heads_per_layer",
                    ),
                )
            model_execution = ModelExecutionDescriptor(
                embedding_scale_q32=_integer(
                    descriptor["embedding_scale_q32"],
                    "model_execution.embedding_scale_q32",
                ),
                final_norm_weight_f16=_variable_hex(
                    descriptor["final_norm_weight_f16"],
                    "model_execution.final_norm_weight_f16",
                ),
                final_norm_epsilon_q32=_integer(
                    descriptor["final_norm_epsilon_q32"],
                    "model_execution.final_norm_epsilon_q32",
                ),
                parameter_root=_hex(
                    descriptor["parameter_root"],
                    32,
                    "model_execution.parameter_root",
                ),
                audit_policy=audit_policy,
            )
        except ManifestFormatError as exc:
            raise ManifestDocumentError(
                "model_execution body is not canonical"
            ) from exc

    operations = []
    operation_fields = {
        "layer",
        "operation_id",
        "expert_id",
        "rows",
        "cols",
        "commitment",
        "weight_scale_q32",
        "min_x_scale_q32",
        "runtime_abs_tolerance_q32",
        "runtime_rel_tolerance_bps",
    }
    for index, raw in enumerate(operations_raw):
        if not isinstance(raw, dict):
            raise ManifestDocumentError(f"operations[{index}] must be an object")
        raw_fields = set(raw)
        if raw_fields == operation_fields:
            weight_block_scales_q32 = ()
        elif raw_fields == operation_fields | {"weight_block_scales_q32"}:
            weight_block_scales_q32 = tuple(
                _integer(
                    scale,
                    f"operations[{index}].weight_block_scales_q32[{scale_index}]",
                )
                for scale_index, scale in enumerate(
                    _list(
                        raw["weight_block_scales_q32"],
                        f"operations[{index}].weight_block_scales_q32",
                    )
                )
            )
        else:
            raise ManifestDocumentError(f"operations[{index}] fields are not canonical")
        operation = _object(raw, raw_fields, f"operations[{index}]")
        expert_id = operation["expert_id"]
        if expert_id is not None:
            expert_id = _integer(expert_id, f"operations[{index}].expert_id")
        try:
            operations.append(
                OperationDescriptor(
                    layer=_integer(
                        operation["layer"],
                        f"operations[{index}].layer",
                        signed=True,
                    ),
                    operation_id=_text(
                        operation["operation_id"],
                        f"operations[{index}].operation_id",
                    ),
                    expert_id=expert_id,
                    rows=_integer(operation["rows"], f"operations[{index}].rows"),
                    cols=_integer(operation["cols"], f"operations[{index}].cols"),
                    commitment=_hex(
                        operation["commitment"],
                        32,
                        f"operations[{index}].commitment",
                    ),
                    weight_scale_q32=_integer(
                        operation["weight_scale_q32"],
                        f"operations[{index}].weight_scale_q32",
                    ),
                    weight_block_scales_q32=weight_block_scales_q32,
                    min_x_scale_q32=_integer(
                        operation["min_x_scale_q32"],
                        f"operations[{index}].min_x_scale_q32",
                    ),
                    runtime_abs_tolerance_q32=_integer(
                        operation["runtime_abs_tolerance_q32"],
                        f"operations[{index}].runtime_abs_tolerance_q32",
                    ),
                    runtime_rel_tolerance_bps=_integer(
                        operation["runtime_rel_tolerance_bps"],
                        f"operations[{index}].runtime_rel_tolerance_bps",
                    ),
                )
            )
        except ManifestFormatError as exc:
            raise ManifestDocumentError(
                f"operations[{index}] body is not canonical"
            ) from exc
    try:
        return StaticWeightCommitmentManifest(
            chain_id=_integer(manifest_value["chain_id"], "chain_id"),
            netuid=_integer(manifest_value["netuid"], "netuid"),
            registry_address=_hex(
                manifest_value["registry_address"], 20, "registry_address"
            ),
            model_spec=spec,
            pcs_suite=_text(manifest_value["pcs_suite"], "pcs_suite"),
            pcs_generator_version=_text(
                manifest_value["pcs_generator_version"],
                "pcs_generator_version",
            ),
            operations=tuple(operations),
            execution_profile=execution_profile,
            layer_execution=tuple(layer_execution),
            model_execution=model_execution,
            protocol_version=_integer(
                manifest_value["protocol_version"], "protocol_version"
            ),
        )
    except ManifestFormatError as exc:
        raise ManifestDocumentError("manifest body is not canonical") from exc


@dataclass(frozen=True)
class SignedManifestDocument:
    manifest: StaticWeightCommitmentManifest
    signatures: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, StaticWeightCommitmentManifest):
            raise ManifestDocumentError("manifest has an unexpected type")
        signatures = tuple(self.signatures)
        if not signatures or len(signatures) > MAX_SIGNATURES:
            raise ManifestDocumentError("manifest signature count is out of range")
        if any(
            not isinstance(signature, bytes) or len(signature) != 65
            for signature in signatures
        ):
            raise ManifestDocumentError("manifest signatures must be exactly 65 bytes")
        if len(signatures) != len(set(signatures)):
            raise ManifestDocumentError(
                "manifest signatures must not contain duplicates"
            )
        object.__setattr__(self, "signatures", signatures)

    def to_dict(self) -> dict:
        return {
            "manifest": manifest_to_dict(self.manifest),
            "signatures": [signature.hex() for signature in self.signatures],
        }

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_MANIFEST_DOCUMENT_BYTES:
            raise ManifestDocumentError("manifest document exceeds the protocol limit")
        return encoded

    @classmethod
    def from_dict(cls, value: object) -> "SignedManifestDocument":
        document = _object(value, {"manifest", "signatures"}, "document")
        manifest = manifest_from_dict(document["manifest"])
        signatures_raw = _list(document["signatures"], "signatures")
        signatures = tuple(
            _hex(item, 65, f"signatures[{index}]")
            for index, item in enumerate(signatures_raw)
        )
        return cls(manifest, signatures)

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "SignedManifestDocument":
        if (
            not isinstance(encoded, bytes)
            or not encoded
            or len(encoded) > MAX_MANIFEST_DOCUMENT_BYTES
        ):
            raise ManifestDocumentError("manifest document length is out of range")
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object_pairs,
                parse_constant=_reject_json_constant,
            )
        except ManifestDocumentError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestDocumentError("manifest document is not valid JSON") from exc
        result = cls.from_dict(value)
        if result.canonical_json_bytes() != encoded:
            raise ManifestDocumentError("manifest document is not canonical JSON")
        return result


def _object(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ManifestDocumentError(f"{name} fields do not match the canonical schema")
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ManifestDocumentError(f"{name} must be a list")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestDocumentError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, signed: bool = False) -> int:
    if type(value) is not int or (not signed and value < 0):
        qualifier = "integer" if signed else "unsigned integer"
        raise ManifestDocumentError(f"{name} must be an {qualifier}")
    return value


def _hex(value: object, size: int, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != size * 2 or value.lower() != value:
        raise ManifestDocumentError(f"{name} must be canonical lowercase hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ManifestDocumentError(f"{name} must be hexadecimal") from exc
    if len(decoded) != size:
        raise ManifestDocumentError(f"{name} must be exactly {size} bytes")
    return decoded


def _variable_hex(value: object, name: str, *, allow_empty: bool = False) -> bytes:
    if (
        not isinstance(value, str)
        or value.lower() != value
        or len(value) % 2
        or (not allow_empty and not value)
    ):
        raise ManifestDocumentError(f"{name} must be canonical lowercase hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ManifestDocumentError(f"{name} must be hexadecimal") from exc
    if not allow_empty and not decoded:
        raise ManifestDocumentError(f"{name} must not be empty")
    return decoded


def load_signed_manifest_document(path: str | Path) -> SignedManifestDocument:
    manifest_path = Path(path).expanduser().resolve()
    try:
        size = manifest_path.stat().st_size
        if size <= 0 or size > MAX_MANIFEST_DOCUMENT_BYTES:
            raise ManifestDocumentError("manifest document file size is out of range")
        with manifest_path.open("rb") as handle:
            encoded = handle.read(MAX_MANIFEST_DOCUMENT_BYTES + 1)
        if len(encoded) != size:
            raise ManifestDocumentError(
                "manifest document file size changed while loading"
            )
    except ManifestDocumentError:
        raise
    except OSError as exc:
        raise ManifestDocumentError(
            f"cannot read manifest document: {manifest_path}"
        ) from exc
    return SignedManifestDocument.from_json_bytes(encoded)


def verify_manifest_document(
    document: SignedManifestDocument,
    *,
    expected_chain_id: int,
    expected_netuid: int,
    expected_registry_address: str | bytes,
    expected_model_spec,
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
) -> tuple[str, ...]:
    return verify_signed_manifest(
        document.manifest,
        document.signatures,
        expected_chain_id=expected_chain_id,
        expected_netuid=expected_netuid,
        expected_registry_address=expected_registry_address,
        expected_model_spec=expected_model_spec,
        expected_pcs_suite=PCS_SUITE,
        expected_pcs_generator_version=PCS_GENERATOR_VERSION,
        expected_operations=document.manifest.operations,
        expected_authority_signers=expected_authority_signers,
        authority_threshold=authority_threshold,
    )


__all__ = [
    "ManifestDocumentError",
    "PCS_GENERATOR_VERSION",
    "PCS_SUITE",
    "SignedManifestDocument",
    "load_signed_manifest_document",
    "manifest_from_dict",
    "manifest_to_dict",
    "model_spec_identity_from_dict",
    "model_spec_identity_to_dict",
    "verify_manifest_document",
]
