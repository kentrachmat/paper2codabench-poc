"""
Verification Seal System - Blockchain-lite approach for result verification.

Provides cryptographic seals for:
- Bundle generation
- Evaluation runs
- Results verification

Uses RSA signatures and SHA-256 hashing for tamper-proof verification.
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class VerificationSeal:
    """
    Handles creation and verification of cryptographic seals.

    Seals prove:
    - When the bundle/evaluation was created (timestamp)
    - What the results were (scores)
    - That the bundle hasn't been tampered with (hash + signature)
    """

    def __init__(self, key_dir: Optional[Path] = None):
        """
        Initialize VerificationSeal.

        Args:
            key_dir: Directory to store/load key pair (default: project root/.keys)
        """
        if key_dir is None:
            from config import Config
            key_dir = Config.PROJECT_ROOT / ".keys"

        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(parents=True, exist_ok=True)

        self.private_key_path = self.key_dir / "private_key.pem"
        self.public_key_path = self.key_dir / "public_key.pem"

        # Load or generate keys
        self._load_or_generate_keys()

    def _load_or_generate_keys(self):
        """Load existing keys or generate new key pair"""
        if self.private_key_path.exists() and self.public_key_path.exists():
            # Load existing keys
            print(f"🔑 Loading existing keys from {self.key_dir}")
            with open(self.private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )

            with open(self.public_key_path, "rb") as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
        else:
            # Generate new key pair
            print(f"🔑 Generating new RSA key pair in {self.key_dir}")
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()

            # Save keys
            with open(self.private_key_path, "wb") as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            with open(self.public_key_path, "wb") as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))

            print("✓ Keys generated and saved")

    def _hash_bundle(self, bundle_path: Path) -> str:
        """
        Hash all files in a bundle directory.

        Args:
            bundle_path: Path to bundle directory

        Returns:
            Hex string of SHA-256 hash
        """
        hasher = hashlib.sha256()

        # Walk through all files in sorted order for consistency
        files_to_hash = []
        for root, dirs, files in os.walk(bundle_path):
            # Skip seals directory to avoid circular dependency
            if 'seals' in Path(root).parts:
                continue

            for file in sorted(files):
                file_path = Path(root) / file
                files_to_hash.append(file_path)

        # Hash each file
        for file_path in sorted(files_to_hash):
            try:
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            except Exception as e:
                print(f"⚠️  Warning: Could not hash {file_path}: {e}")

        return hasher.hexdigest()

    def _hash_dict(self, data: dict) -> str:
        """
        Create deterministic hash of dictionary.

        Args:
            data: Dictionary to hash

        Returns:
            Hex string of SHA-256 hash
        """
        # Sort keys for deterministic JSON
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def create_seal(
        self,
        seal_type: str,
        bundle_id: str,
        data: Dict[str, Any],
        bundle_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a verification seal.

        Args:
            seal_type: Type of seal ('bundle_creation', 'evaluation_run', etc.)
            bundle_id: Identifier for the bundle
            data: Data to include in seal (e.g., scores, metadata)
            bundle_path: Path to bundle (if applicable) for hashing

        Returns:
            Seal dictionary with signature
        """
        # Build seal data
        seal_data = {
            "seal_type": seal_type,
            "bundle_id": bundle_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data,
        }

        # Add bundle hash if path provided
        if bundle_path:
            seal_data["bundle_hash"] = self._hash_bundle(bundle_path)

        # Create canonical representation for signing
        data_to_sign = {k: seal_data[k] for k in sorted(seal_data.keys())}
        data_bytes = json.dumps(data_to_sign, sort_keys=True).encode()

        # Sign the data
        signature = self.private_key.sign(
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Add signature to seal
        seal_data["signature"] = signature.hex()
        seal_data["seal_id"] = self._hash_dict(seal_data)[:16]  # Short ID

        return seal_data

    def verify_seal(self, seal_data: Dict[str, Any]) -> bool:
        """
        Verify a seal's signature.

        Args:
            seal_data: Seal dictionary with signature

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Extract and remove signature
            signature_hex = seal_data.pop("signature")
            seal_id = seal_data.pop("seal_id", None)  # Remove seal_id too

            signature = bytes.fromhex(signature_hex)

            # Recreate canonical representation
            data_to_verify = {k: seal_data[k] for k in sorted(seal_data.keys())}
            data_bytes = json.dumps(data_to_verify, sort_keys=True).encode()

            # Verify signature
            self.public_key.verify(
                signature,
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            # Restore signature and seal_id for later use
            seal_data["signature"] = signature_hex
            if seal_id:
                seal_data["seal_id"] = seal_id

            return True

        except Exception as e:
            # Restore signature even on failure
            if "signature_hex" in locals():
                seal_data["signature"] = signature_hex
            if "seal_id" in locals() and seal_id:
                seal_data["seal_id"] = seal_id

            print(f"⚠️  Verification failed: {e}")
            return False

    def save_seal(self, seal_data: Dict[str, Any], output_path: Path):
        """
        Save seal to JSON file.

        Args:
            seal_data: Seal dictionary
            output_path: Path to save seal JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(seal_data, f, indent=2)

        print(f"💾 Seal saved to {output_path}")
        print(f"   Seal ID: {seal_data.get('seal_id', 'N/A')}")

    def load_seal(self, seal_path: Path) -> Dict[str, Any]:
        """
        Load seal from JSON file.

        Args:
            seal_path: Path to seal JSON file

        Returns:
            Seal dictionary
        """
        with open(seal_path, 'r') as f:
            return json.load(f)


# Convenience functions
def create_bundle_seal(bundle_path: Path, metadata: dict) -> Dict[str, Any]:
    """
    Create a seal for bundle generation.

    Args:
        bundle_path: Path to bundle directory
        metadata: Bundle metadata (task_name, etc.)

    Returns:
        Seal dictionary
    """
    seal = VerificationSeal()
    bundle_id = bundle_path.name

    seal_data = seal.create_seal(
        seal_type="bundle_creation",
        bundle_id=bundle_id,
        data=metadata,
        bundle_path=bundle_path
    )

    # Save seal to bundle/seals/
    seals_dir = bundle_path / "seals"
    seals_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    seal_path = seals_dir / f"bundle_creation_{timestamp}.json"
    seal.save_seal(seal_data, seal_path)

    return seal_data


def create_evaluation_seal(bundle_path: Path, scores: dict, submission_info: dict) -> Dict[str, Any]:
    """
    Create a seal for an evaluation run.

    Args:
        bundle_path: Path to bundle directory
        scores: Evaluation scores dictionary
        submission_info: Information about the submission

    Returns:
        Seal dictionary
    """
    seal = VerificationSeal()
    bundle_id = bundle_path.name

    data = {
        "scores": scores,
        "submission": submission_info
    }

    seal_data = seal.create_seal(
        seal_type="evaluation_run",
        bundle_id=bundle_id,
        data=data,
        bundle_path=bundle_path
    )

    # Save seal to bundle/seals/
    seals_dir = bundle_path / "seals"
    seals_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    seal_path = seals_dir / f"evaluation_{timestamp}.json"
    seal.save_seal(seal_data, seal_path)

    return seal_data


def verify_seal_file(seal_path: Path) -> bool:
    """
    Verify a seal from file.

    Args:
        seal_path: Path to seal JSON file

    Returns:
        True if valid, False otherwise
    """
    seal = VerificationSeal()
    seal_data = seal.load_seal(seal_path)
    return seal.verify_seal(seal_data)
