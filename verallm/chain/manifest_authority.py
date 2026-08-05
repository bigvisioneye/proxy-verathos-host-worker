"""Read the proof manifest authority from the model registry owner."""

from __future__ import annotations

from dataclasses import dataclass


_MULTISIG_VIEW_ABI = (
    {
        "inputs": [],
        "name": "getSigners",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "threshold",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
)


def _address(value: str) -> str:
    if not isinstance(value, str) or len(value) != 42 or not value.startswith("0x"):
        raise ValueError("manifest authority must be a 20-byte hexadecimal address")
    try:
        bytes.fromhex(value[2:])
    except ValueError as exc:
        raise ValueError("manifest authority must be hexadecimal") from exc
    return value.lower()


@dataclass(frozen=True)
class ManifestAuthority:
    """Current signer set and threshold for static proof manifests."""

    owner: str
    signers: tuple[str, ...]
    threshold: int

    def __post_init__(self) -> None:
        owner = _address(self.owner)
        signers = tuple(sorted(_address(value) for value in self.signers))
        if not signers or len(signers) != len(set(signers)):
            raise ValueError("manifest authority signers must be non-empty and unique")
        if isinstance(self.threshold, bool) or not 1 <= self.threshold <= len(signers):
            raise ValueError("manifest authority threshold is out of range")
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "signers", signers)


def resolve_manifest_authority(model_registry_client) -> ManifestAuthority:
    """Resolve an EOA owner or the configured Verathos multisig signer set."""

    from web3 import Web3

    provider = model_registry_client._provider
    registry = model_registry_client._contract
    owner = _address(provider.call_with_retry(lambda: registry.functions.owner().call()))
    owner_checksum = Web3.to_checksum_address(owner)
    owner_code = bytes(
        provider.call_with_retry(lambda: provider.w3.eth.get_code(owner_checksum))
    )
    if not owner_code:
        return ManifestAuthority(owner=owner, signers=(owner,), threshold=1)

    multisig = provider.w3.eth.contract(
        address=owner_checksum,
        abi=list(_MULTISIG_VIEW_ABI),
    )
    try:
        raw_signers = provider.call_with_retry(
            lambda: multisig.functions.getSigners().call()
        )
        threshold = provider.call_with_retry(
            lambda: multisig.functions.threshold().call()
        )
    except Exception as exc:
        raise RuntimeError(
            "model registry owner contract does not expose the expected manifest authority views"
        ) from exc
    return ManifestAuthority(
        owner=owner,
        signers=tuple(raw_signers),
        threshold=int(threshold),
    )
