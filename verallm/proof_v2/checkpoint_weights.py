"""Canonical miner-side weight witnesses for the supported proof-v2 profile.

The signed catalog contains PCS commitments, not the private prover witness.
Miners therefore reconstruct the exact signed-int8 witness from their local
checkpoint and cache it by manifest digest.  This module deliberately supports
only the checkpoint layout accepted by the Qwen dense execution-profile
artifact generator; unknown layouts fail closed instead of guessing how a
runtime quantization wrapper represents its weights.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from verallm.challenge.v2 import (
    MODEL_LM_HEAD_OPERATION_ID,
    MODEL_OPERATION_LAYER_IDX,
    OperationKeyV2,
)
from verallm.proof_v2.layout import (
    FULL_OUTPUT_OPERATION_ID,
    FULL_QKV_OPERATION_ID,
    GDN_BA_OPERATION_ID,
    GDN_OUTPUT_OPERATION_ID,
    GDN_QKVZ_OPERATION_ID,
    MAX_BLOCK_AXIS,
    MLP_DOWN_OPERATION_ID,
    MLP_GATE_UP_OPERATION_ID,
    operation_descriptor_by_key,
    registered_all_operations_from_manifest,
    validate_qwen_hybrid_execution_manifest_profile,
)
from verallm.proof_v2.manifest import StaticWeightCommitmentManifest

CHECKPOINT_DIRECTORY_ENV = "VERATHOS_PROOF_V2_CHECKPOINT_DIR"

_MAX_CONFIG_BYTES = 16 << 20
_MAX_INDEX_BYTES = 64 << 20
_MAX_SAFETENSORS_BYTES = 256 << 30
_PACK_BITS = 4
_PACK_FACTOR = 32 // _PACK_BITS
_SUPPORTED_LAYER_PREFIXES = (
    "model.language_model.layers",
    "model.layers",
)
_REQUIRED_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PACKED_SUFFIXES = (
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
)


class ProofV2CheckpointWeightError(RuntimeError):
    """The local checkpoint cannot reproduce an authenticated W witness."""


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProofV2CheckpointWeightError(
                "checkpoint JSON contains a duplicate key"
            )
        result[key] = value
    return result


def _read_json(path: Path, *, maximum: int, name: str) -> dict:
    try:
        size = path.stat().st_size
        if not 0 < size <= maximum or not path.is_file():
            raise ProofV2CheckpointWeightError(f"{name} size is invalid")
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ProofV2CheckpointWeightError(
                    f"{name} contains a non-finite JSON number"
                )
            ),
        )
    except ProofV2CheckpointWeightError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofV2CheckpointWeightError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProofV2CheckpointWeightError(f"{name} must be a JSON object")
    return value


def _checkpoint_member(checkpoint: Path, relative: str, *, name: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ProofV2CheckpointWeightError(f"{name} path is invalid")
    candidate = (checkpoint / relative).resolve()
    try:
        candidate.relative_to(checkpoint)
    except ValueError as exc:
        raise ProofV2CheckpointWeightError(f"{name} escapes the checkpoint") from exc
    if not candidate.is_file():
        raise ProofV2CheckpointWeightError(f"{name} is missing")
    return candidate


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProofV2CheckpointWeightError(f"{name} must be a positive integer")
    return value


def _text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    nested = config.get("text_config")
    if nested is None:
        return config
    if not isinstance(nested, dict):
        raise ProofV2CheckpointWeightError("checkpoint text_config is malformed")
    return nested


def _quantization_group_size(config: Mapping[str, object]) -> int:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ProofV2CheckpointWeightError("checkpoint quantization_config is missing")
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("format") != "pack-quantized"
        or quantization.get("quantization_status") != "compressed"
        or quantization.get("sparsity_config", {}) not in ({}, None)
        or quantization.get("transform_config", {}) not in ({}, None)
    ):
        raise ProofV2CheckpointWeightError(
            "checkpoint is not the supported dense pack-quantized format"
        )
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        raise ProofV2CheckpointWeightError("checkpoint quantization group is ambiguous")
    group = next(iter(groups.values()))
    if not isinstance(group, dict):
        raise ProofV2CheckpointWeightError("checkpoint quantization group is malformed")
    weights = group.get("weights")
    if (
        group.get("format") != "pack-quantized"
        or group.get("targets") != ["Linear"]
        or group.get("input_activations") is not None
        or group.get("output_activations") is not None
        or not isinstance(weights, dict)
        or weights.get("type") != "int"
        or weights.get("num_bits") != _PACK_BITS
        or weights.get("strategy") != "group"
        or weights.get("symmetric") is not False
        or weights.get("dynamic") is not False
        or weights.get("actorder") is not None
        or weights.get("block_structure") is not None
        or weights.get("scale_dtype") is not None
        or weights.get("zp_dtype") != "torch.int8"
    ):
        raise ProofV2CheckpointWeightError(
            "checkpoint weight quantization parameters are unsupported"
        )
    return _positive_integer(weights.get("group_size"), "quantization group_size")


def _resolve_checkpoint(model_id: str, configured: str | Path | None) -> Path:
    raw = str(configured or os.environ.get(CHECKPOINT_DIRECTORY_ENV, "")).strip()
    if raw:
        checkpoint = Path(raw).expanduser().resolve()
    else:
        model_path = Path(model_id).expanduser()
        if model_path.is_dir():
            checkpoint = model_path.resolve()
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ProofV2CheckpointWeightError(
                    "local proof-v2 witness reconstruction requires huggingface_hub"
                ) from exc
            try:
                checkpoint = Path(
                    snapshot_download(model_id, local_files_only=True)
                ).resolve()
            except Exception as exc:
                raise ProofV2CheckpointWeightError(
                    "the exact proof-v2 checkpoint is not available locally"
                ) from exc
    if not checkpoint.is_dir():
        raise ProofV2CheckpointWeightError(
            "proof-v2 checkpoint must be an existing directory"
        )
    return checkpoint


class _TensorStore:
    def __init__(self, checkpoint: Path, weight_map: Mapping[str, str]):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ProofV2CheckpointWeightError(
                "proof-v2 witness reconstruction requires safetensors"
            ) from exc
        self._checkpoint = checkpoint
        self._weight_map = weight_map
        self._safe_open = safe_open
        self._stack = ExitStack()
        self._files = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._stack.__exit__(exc_type, exc, traceback)

    def _file(self, key: str):
        filename = self._weight_map.get(key)
        if not isinstance(filename, str):
            raise ProofV2CheckpointWeightError(f"checkpoint tensor {key!r} is missing")
        handle = self._files.get(filename)
        if handle is not None:
            return handle
        path = _checkpoint_member(
            self._checkpoint,
            filename,
            name="checkpoint safetensors shard",
        )
        if path.stat().st_size > _MAX_SAFETENSORS_BYTES:
            raise ProofV2CheckpointWeightError(
                "checkpoint safetensors shard exceeds the supported size"
            )
        handle = self._stack.enter_context(
            self._safe_open(str(path), framework="pt", device="cpu")
        )
        self._files[filename] = handle
        return handle

    def tensor(self, key: str):
        try:
            return self._file(key).get_tensor(key)
        except ProofV2CheckpointWeightError:
            raise
        except Exception as exc:
            raise ProofV2CheckpointWeightError(
                f"checkpoint tensor {key!r} cannot be read"
            ) from exc

    def slice(self, key: str):
        try:
            return self._file(key).get_slice(key)
        except ProofV2CheckpointWeightError:
            raise
        except Exception as exc:
            raise ProofV2CheckpointWeightError(
                f"checkpoint tensor {key!r} cannot be sliced"
            ) from exc


def _unpack_signed_int4_rows(packed, *, columns: int):
    import torch

    shifts = torch.arange(_PACK_FACTOR, dtype=torch.int32) * _PACK_BITS
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(
        packed.shape[0], packed.shape[1] * _PACK_FACTOR
    )
    return (unpacked[:, :columns] - 8).to(torch.int8)


def _unpack_signed_zero_points(packed, *, rows: int):
    import torch

    shifts = torch.arange(_PACK_FACTOR, dtype=torch.int32) * _PACK_BITS
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1)
    return (
        unpacked.reshape(packed.shape[0] * _PACK_FACTOR, packed.shape[1])[:rows] - 8
    ).to(torch.int8)


class CanonicalCheckpointWeightProviderV2:
    """Reconstruct and cache exact challenged W blocks for one manifest."""

    def __init__(
        self,
        *,
        model_id: str,
        manifest: StaticWeightCommitmentManifest,
        checkpoint_directory: str | Path | None = None,
    ):
        if not isinstance(manifest, StaticWeightCommitmentManifest):
            raise ProofV2CheckpointWeightError("manifest has an unexpected type")
        validate_qwen_hybrid_execution_manifest_profile(manifest)
        if manifest.model_spec.model_id != model_id:
            raise ProofV2CheckpointWeightError(
                "miner model id does not match the proof-v2 manifest"
            )
        if any(
            not descriptor.weight_block_scales_q32 for descriptor in manifest.operations
        ):
            raise ProofV2CheckpointWeightError(
                "canonical checkpoint witnesses require block-scaled operations"
            )
        self.manifest = manifest
        self.checkpoint = _resolve_checkpoint(model_id, checkpoint_directory)
        self.descriptors = operation_descriptor_by_key(manifest)
        self.operations = {
            operation.key: operation
            for operation in registered_all_operations_from_manifest(manifest)
        }
        self._locks_guard = threading.Lock()
        self._locks = {}
        self._block_cache = OrderedDict()
        self._block_cache_bytes = 0
        self._block_cache_limit = 512 << 20

        config_path = _checkpoint_member(
            self.checkpoint, "config.json", name="checkpoint config"
        )
        self.config = _read_json(
            config_path, maximum=_MAX_CONFIG_BYTES, name="checkpoint config"
        )
        index_names = tuple(
            name
            for name in (
                "model.safetensors.index.json",
                "consolidated.safetensors.index.json",
            )
            if (self.checkpoint / name).exists()
        )
        if len(index_names) != 1:
            raise ProofV2CheckpointWeightError(
                "checkpoint must contain one supported safetensors index"
            )
        index = _read_json(
            _checkpoint_member(
                self.checkpoint,
                index_names[0],
                name="checkpoint safetensors index",
            ),
            maximum=_MAX_INDEX_BYTES,
            name="checkpoint safetensors index",
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or any(
            type(key) is not str or type(value) is not str
            for key, value in weight_map.items()
        ):
            raise ProofV2CheckpointWeightError(
                "checkpoint index weight_map is malformed"
            )
        self.weight_map = weight_map
        self._validate_layout()

    def _validate_layout(self) -> None:
        spec = self.manifest.model_spec
        if (
            spec.quant_mode != "int4"
            or spec.num_experts != 0
            or spec.expert_w_num_cols != 0
        ):
            raise ProofV2CheckpointWeightError(
                "checkpoint witness provider supports dense int4 ModelSpecs only"
            )
        text = _text_config(self.config)
        hidden = _positive_integer(text.get("hidden_size"), "checkpoint hidden_size")
        intermediate = _positive_integer(
            text.get("intermediate_size"), "checkpoint intermediate_size"
        )
        layers = _positive_integer(
            text.get("num_hidden_layers"), "checkpoint num_hidden_layers"
        )
        if (
            hidden != spec.hidden_dim
            or intermediate * 2 != spec.intermediate_dim
            or layers != spec.num_layers
        ):
            raise ProofV2CheckpointWeightError(
                "checkpoint dimensions do not match the signed ModelSpec"
            )
        self.group_size = _quantization_group_size(self.config)
        if hidden % _PACK_FACTOR or hidden % self.group_size:
            raise ProofV2CheckpointWeightError(
                "checkpoint dimensions are incompatible with its INT4 groups"
            )
        candidates = []
        for prefix in _SUPPORTED_LAYER_PREFIXES:
            if all(
                f"{prefix}.{layer}.mlp.{projection}.{suffix}" in self.weight_map
                for layer in range(layers)
                for projection in _REQUIRED_MLP_PROJECTIONS
                for suffix in _PACKED_SUFFIXES
            ):
                candidates.append(prefix)
        if len(candidates) != 1:
            raise ProofV2CheckpointWeightError(
                "checkpoint does not contain one exact supported layer prefix"
            )
        self.layer_prefix = candidates[0]
        layer_types = text.get("layer_types")
        if (
            self.config.get("model_type") not in ("qwen3_5", "qwen3_6")
            or not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(
                item not in ("full_attention", "linear_attention")
                for item in layer_types
            )
        ):
            raise ProofV2CheckpointWeightError(
                "checkpoint is not the supported dense Qwen hybrid profile"
            )
        self.layer_types = tuple(layer_types)

    def _lock_for(self, key: OperationKeyV2) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _parts(self, key: OperationKeyV2) -> tuple[str, tuple[str, ...]]:
        if key.layer_idx == MODEL_OPERATION_LAYER_IDX:
            if key.operation_id != MODEL_LM_HEAD_OPERATION_ID:
                raise ProofV2CheckpointWeightError(
                    "unknown model-level proof-v2 operation"
                )
            return "float", ("lm_head.weight",)
        if not 0 <= key.layer_idx < len(self.layer_types):
            raise ProofV2CheckpointWeightError("operation layer is out of range")
        prefix = f"{self.layer_prefix}.{key.layer_idx}"
        operation = key.operation_id
        if operation == MLP_GATE_UP_OPERATION_ID:
            return "packed", tuple(
                f"{prefix}.mlp.{name}" for name in ("gate_proj", "up_proj")
            )
        if operation == MLP_DOWN_OPERATION_ID:
            return "packed", (f"{prefix}.mlp.down_proj",)
        layer_type = self.layer_types[key.layer_idx]
        if layer_type == "full_attention":
            if operation == FULL_QKV_OPERATION_ID:
                return "packed", tuple(
                    f"{prefix}.self_attn.{name}_proj" for name in ("q", "k", "v")
                )
            if operation == FULL_OUTPUT_OPERATION_ID:
                return "packed", (f"{prefix}.self_attn.o_proj",)
        else:
            if operation == GDN_QKVZ_OPERATION_ID:
                return "packed", (
                    f"{prefix}.linear_attn.in_proj_qkv",
                    f"{prefix}.linear_attn.in_proj_z",
                )
            if operation == GDN_BA_OPERATION_ID:
                return "float", (
                    f"{prefix}.linear_attn.in_proj_b.weight",
                    f"{prefix}.linear_attn.in_proj_a.weight",
                )
            if operation == GDN_OUTPUT_OPERATION_ID:
                return "packed", (f"{prefix}.linear_attn.out_proj",)
        raise ProofV2CheckpointWeightError(
            "operation is not part of the checkpoint layer profile"
        )

    def _packed_projection(self, store: _TensorStore, base: str):
        import torch

        shape_tensor = store.tensor(f"{base}.weight_shape")
        if shape_tensor.dtype != torch.int64 or tuple(shape_tensor.shape) != (2,):
            raise ProofV2CheckpointWeightError(f"{base} weight_shape is malformed")
        shape = tuple(int(item) for item in shape_tensor.tolist())
        if len(shape) != 2 or min(shape) <= 0:
            raise ProofV2CheckpointWeightError(f"{base} weight_shape is malformed")
        packed = store.slice(f"{base}.weight_packed")
        scale = store.slice(f"{base}.weight_scale")
        zero = store.slice(f"{base}.weight_zero_point")
        groups = shape[1] // self.group_size
        if (
            shape[1] % self.group_size
            or packed.get_dtype() != "I32"
            or tuple(packed.get_shape()) != (shape[0], shape[1] // _PACK_FACTOR)
            or scale.get_dtype() != "F16"
            or tuple(scale.get_shape()) != (shape[0], groups)
            or zero.get_dtype() != "I32"
            or tuple(zero.get_shape()) != (shape[0] // _PACK_FACTOR, groups)
        ):
            raise ProofV2CheckpointWeightError(
                f"{base} packed projection layout is unsupported"
            )
        return packed, scale, zero, shape

    @staticmethod
    def _quantize_selected_block(values, *, expected_scale_q32: int) -> np.ndarray:
        """Canonicalize one complete manifest scale block to ``[K, cols]``."""

        import torch

        if values.dim() != 2 or not 0 < int(values.shape[0]) <= MAX_BLOCK_AXIS:
            raise ProofV2CheckpointWeightError(
                "selected source weight block has invalid dimensions"
            )
        block = values.float()
        maximum = float(block.abs().max().item())
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ProofV2CheckpointWeightError(
                "selected source weight block is non-finite"
            )
        if maximum == 0.0:
            quantized = torch.zeros_like(block, dtype=torch.int8)
            scale_q32 = 1
        else:
            quantized = (
                (block / maximum * 127.0).round().clamp(-128, 127).to(torch.int8)
            )
            scale_q32 = int(round(maximum / 127.0 * float(1 << 32)))
        if scale_q32 != expected_scale_q32:
            raise ProofV2CheckpointWeightError(
                "local checkpoint block scale does not match the signed manifest"
            )
        return np.ascontiguousarray(quantized.T.numpy(), dtype=np.int8)

    def _packed_selected_block(
        self,
        store: _TensorStore,
        bases: Sequence[str],
        *,
        column_offset: int,
        columns: int,
        expected_scale_q32: int,
    ) -> np.ndarray:
        import torch

        projections = tuple(self._packed_projection(store, base) for base in bases)
        input_dim = projections[0][3][1]
        if any(item[3][1] != input_dim for item in projections):
            raise ProofV2CheckpointWeightError(
                "fused packed projections have different input dimensions"
            )
        requested_stop = column_offset + columns
        part_offset = 0
        pieces = []
        for packed, scale, zero, shape in projections:
            part_stop = part_offset + shape[0]
            overlap_start = max(column_offset, part_offset)
            overlap_stop = min(requested_stop, part_stop)
            if overlap_start < overlap_stop:
                local_start = overlap_start - part_offset
                local_stop = overlap_stop - part_offset
                packed_rows = packed[local_start:local_stop]
                scale_rows = scale[local_start:local_stop]
                zero_packed_start = local_start // _PACK_FACTOR
                zero_packed_stop = (local_stop + _PACK_FACTOR - 1) // _PACK_FACTOR
                zero_rows = _unpack_signed_zero_points(
                    zero[zero_packed_start:zero_packed_stop],
                    rows=(zero_packed_stop - zero_packed_start) * _PACK_FACTOR,
                )
                zero_offset = local_start - zero_packed_start * _PACK_FACTOR
                zero_rows = zero_rows[
                    zero_offset : zero_offset + local_stop - local_start
                ]
                if (
                    packed_rows.dtype != torch.int32
                    or scale_rows.dtype != torch.float16
                    or zero_rows.dtype != torch.int8
                    or not bool(torch.isfinite(scale_rows).all())
                    or not bool((scale_rows > 0).all())
                ):
                    raise ProofV2CheckpointWeightError(
                        "selected packed source rows are not canonical"
                    )
                quantized = _unpack_signed_int4_rows(
                    packed_rows, columns=input_dim
                ).reshape(
                    local_stop - local_start,
                    input_dim // self.group_size,
                    self.group_size,
                )
                values = (
                    (
                        quantized.to(scale_rows.dtype)
                        - zero_rows.unsqueeze(-1).to(scale_rows.dtype)
                    )
                    * scale_rows.unsqueeze(-1)
                ).reshape(local_stop - local_start, input_dim)
                pieces.append(values)
            part_offset = part_stop
        if (
            requested_stop > part_offset
            or sum(int(item.shape[0]) for item in pieces) != columns
        ):
            raise ProofV2CheckpointWeightError(
                "selected packed weight block is outside the fused projection"
            )
        values = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        return self._quantize_selected_block(
            values,
            expected_scale_q32=expected_scale_q32,
        )

    def _float_selected_block(
        self,
        store: _TensorStore,
        keys: Sequence[str],
        *,
        column_offset: int,
        columns: int,
        expected_scale_q32: int,
    ) -> np.ndarray:
        import torch

        sources = tuple(store.slice(key) for key in keys)
        shapes = tuple(tuple(source.get_shape()) for source in sources)
        if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
            raise ProofV2CheckpointWeightError("floating projection shape is invalid")
        input_dim = shapes[0][1]
        if any(shape[1] != input_dim for shape in shapes):
            raise ProofV2CheckpointWeightError(
                "fused floating projections have different input dimensions"
            )
        requested_stop = column_offset + columns
        part_offset = 0
        pieces = []
        for source, shape in zip(sources, shapes):
            part_stop = part_offset + shape[0]
            overlap_start = max(column_offset, part_offset)
            overlap_stop = min(requested_stop, part_stop)
            if overlap_start < overlap_stop:
                values = (
                    source[overlap_start - part_offset : overlap_stop - part_offset]
                    .detach()
                    .cpu()
                )
                if values.dtype != torch.float16 or not bool(
                    torch.isfinite(values).all()
                ):
                    raise ProofV2CheckpointWeightError(
                        "floating proof weight is not canonical finite FP16"
                    )
                pieces.append(values)
            part_offset = part_stop
        if (
            requested_stop > part_offset
            or sum(int(item.shape[0]) for item in pieces) != columns
        ):
            raise ProofV2CheckpointWeightError(
                "selected floating weight block is outside the fused projection"
            )
        values = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        return self._quantize_selected_block(
            values,
            expected_scale_q32=expected_scale_q32,
        )

    def _cache_block(
        self,
        cache_key: tuple[OperationKeyV2, int, int],
        matrix: np.ndarray,
    ) -> None:
        size = int(matrix.nbytes)
        while (
            self._block_cache
            and self._block_cache_bytes + size > self._block_cache_limit
        ):
            _, evicted = self._block_cache.popitem(last=False)
            self._block_cache_bytes -= int(evicted.nbytes)
        self._block_cache[cache_key] = matrix
        self._block_cache_bytes += size

    def selected_blocks(
        self,
        key: OperationKeyV2,
        blocks: Sequence[tuple[int, int]],
    ) -> Mapping[tuple[int, int], np.ndarray]:
        """Return only the exact post-challenge W blocks needed by the proof."""

        operation = self.operations.get(key)
        descriptor = self.descriptors.get(key)
        if operation is None or descriptor is None:
            raise ProofV2CheckpointWeightError(
                "weight operation is not authenticated by the manifest"
            )
        requested = tuple(blocks)
        if not requested or len(requested) != len(set(requested)):
            raise ProofV2CheckpointWeightError(
                "selected checkpoint weight block set is not exact"
            )
        for column_offset, columns in requested:
            expected_columns = min(MAX_BLOCK_AXIS, operation.output_dim - column_offset)
            if (
                column_offset < 0
                or column_offset % MAX_BLOCK_AXIS
                or columns != expected_columns
                or columns <= 0
            ):
                raise ProofV2CheckpointWeightError(
                    "selected checkpoint weight block is not canonical"
                )
        result = {}
        missing = []
        with self._locks_guard:
            for column_offset, columns in requested:
                cache_key = (key, column_offset, columns)
                matrix = self._block_cache.get(cache_key)
                if matrix is None:
                    missing.append((column_offset, columns))
                else:
                    self._block_cache.move_to_end(cache_key)
                    result[(column_offset, columns)] = matrix
        if missing:
            encoding, parts = self._parts(key)
            with self._lock_for(key):
                with _TensorStore(self.checkpoint, self.weight_map) as store:
                    for column_offset, columns in missing:
                        cache_key = (key, column_offset, columns)
                        with self._locks_guard:
                            matrix = self._block_cache.get(cache_key)
                        if matrix is None:
                            scale_index = column_offset // MAX_BLOCK_AXIS
                            try:
                                expected_scale = descriptor.weight_block_scales_q32[
                                    scale_index
                                ]
                            except IndexError as exc:
                                raise ProofV2CheckpointWeightError(
                                    "signed weight scale block is missing"
                                ) from exc
                            matrix = (
                                self._packed_selected_block(
                                    store,
                                    parts,
                                    column_offset=column_offset,
                                    columns=columns,
                                    expected_scale_q32=expected_scale,
                                )
                                if encoding == "packed"
                                else self._float_selected_block(
                                    store,
                                    parts,
                                    column_offset=column_offset,
                                    columns=columns,
                                    expected_scale_q32=expected_scale,
                                )
                            )
                            with self._locks_guard:
                                self._cache_block(cache_key, matrix)
                        result[(column_offset, columns)] = matrix
        if set(result) != set(requested):
            raise ProofV2CheckpointWeightError(
                "selected checkpoint weight block set is incomplete"
            )
        return result


__all__ = [
    "CHECKPOINT_DIRECTORY_ENV",
    "CanonicalCheckpointWeightProviderV2",
    "ProofV2CheckpointWeightError",
]
