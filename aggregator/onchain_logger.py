"""
FzIQ On-Chain Logger — opBNB Integration

Writes training round records to the FzIQTrainingLog smart contract.
Each record: {round, agent_ids, scenario_hashes, scores, model_hash_before, model_hash_after}
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Contract ABI (minimal — only the functions we call)
FZIQ_ABI = [
    {
        "inputs": [
            {"name": "modelHashBefore", "type": "bytes32"},
            {"name": "modelHashAfter", "type": "bytes32"},
            {"name": "scenarioHashes", "type": "bytes32[]"},
            {"name": "agentIds", "type": "address[]"},
            {"name": "scores", "type": "uint256[]"},
        ],
        "name": "logRound",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "roundId", "type": "uint256"}, {"name": "expectedHash", "type": "bytes32"}],
        "name": "verifyRound",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "roundId", "type": "uint256"}],
        "name": "getRoundDetails",
        "outputs": [
            {
                "components": [
                    {"name": "roundId", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "numGradients", "type": "uint256"},
                    {"name": "modelHashBefore", "type": "bytes32"},
                    {"name": "modelHashAfter", "type": "bytes32"},
                    {"name": "scenarioHashes", "type": "bytes32[]"},
                    {"name": "agentIds", "type": "address[]"},
                    {"name": "scores", "type": "uint256[]"},
                    {"name": "verified", "type": "bool"},
                ],
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "roundId", "type": "uint256"},
            {"indexed": False, "name": "modelHashBefore", "type": "bytes32"},
            {"indexed": False, "name": "modelHashAfter", "type": "bytes32"},
            {"indexed": False, "name": "numContributors", "type": "uint256"},
        ],
        "name": "RoundLogged",
        "type": "event",
    },
]

SCORE_SCALE = 10_000  # scores stored as uint256 * 1e4


class OnChainLogger:
    """
    Writes FzIQ training records to opBNB blockchain.
    
    Uses the aggregator's private key to send transactions.
    One transaction per 24-hour aggregation cycle (~$0.0001 on opBNB).
    
    Example:
        logger = OnChainLogger()
        tx_hash = logger.log_round(
            model_hash_before="abc123...",
            model_hash_after="def456...",
            scenario_hashes=["hash1", "hash2"],
            agent_ids=["0xABC", "0xDEF"],
            scores=[0.72, 0.45],
        )
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("OPBNB_RPC_URL", "https://opbnb-testnet-rpc.bnbchain.org")
        self.private_key = private_key or os.getenv("AGGREGATOR_PRIVATE_KEY")
        self.contract_address = contract_address or os.getenv("CONTRACT_ADDRESS")

        self._w3 = None
        self._contract = None

    def _connect(self):
        """Lazy connect to web3."""
        if self._w3 is not None:
            return

        try:
            from web3 import Web3
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if not self._w3.is_connected():
                raise ConnectionError(f"Cannot connect to {self.rpc_url}")
            
            if self.contract_address:
                self._contract = self._w3.eth.contract(
                    address=Web3.to_checksum_address(self.contract_address),
                    abi=FZIQ_ABI,
                )
            logger.info(f"Connected to opBNB at {self.rpc_url}")
        except ImportError:
            logger.error("web3 not installed. Run: pip install web3")
            raise

    def _hex_to_bytes32(self, hex_str: str) -> bytes:
        """Convert 64-char hex string to bytes32."""
        clean = hex_str.replace("0x", "").ljust(64, "0")[:64]
        return bytes.fromhex(clean)

    def _agent_id_to_address(self, agent_id: str) -> str:
        """Convert agent_id string to Ethereum address format."""
        from web3 import Web3
        # Hash agent_id to produce deterministic address
        import hashlib
        h = hashlib.sha256(agent_id.encode()).hexdigest()[:40]
        return Web3.to_checksum_address("0x" + h)

    def log_round(
        self,
        model_hash_before: str,
        model_hash_after: str,
        scenario_hashes: List[str],
        agent_ids: List[str],
        scores: List[float],
        dry_run: bool = False,
    ) -> Optional[str]:
        """
        Write a training round record to opBNB.
        
        Args:
            model_hash_before: SHA-256 hex of metamodel before aggregation
            model_hash_after: SHA-256 hex of metamodel after aggregation
            scenario_hashes: list of scenario SHA-256 hashes
            agent_ids: list of agent ID strings
            scores: list of combined scores 0-1
            dry_run: if True, skip actual transaction (for testing)
        Returns:
            Transaction hash string, or None if dry_run
        """
        if dry_run:
            logger.info(
                f"[DRY RUN] Would log round: "
                f"{len(agent_ids)} agents, "
                f"hash {model_hash_before[:8]}→{model_hash_after[:8]}"
            )
            return None

        if not self.private_key:
            raise ValueError("AGGREGATOR_PRIVATE_KEY not set")
        if not self.contract_address:
            raise ValueError("CONTRACT_ADDRESS not set")

        self._connect()
        from web3 import Web3
        from eth_account import Account

        account = Account.from_key(self.private_key)

        # Encode parameters
        hash_before_b32 = self._hex_to_bytes32(model_hash_before)
        hash_after_b32 = self._hex_to_bytes32(model_hash_after)
        scenario_b32 = [self._hex_to_bytes32(h) for h in scenario_hashes]
        addresses = [self._agent_id_to_address(aid) for aid in agent_ids]
        scores_scaled = [int(s * SCORE_SCALE) for s in scores]

        # Build transaction
        nonce = self._w3.eth.get_transaction_count(account.address)
        gas_price = self._w3.eth.gas_price

        tx = self._contract.functions.logRound(
            hash_before_b32,
            hash_after_b32,
            scenario_b32,
            addresses,
            scores_scaled,
        ).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": 500_000,
        })

        signed = account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        logger.info(
            f"Round logged on-chain: tx={tx_hash.hex()}, "
            f"block={receipt['blockNumber']}, "
            f"status={'success' if receipt['status'] == 1 else 'FAILED'}"
        )
        return tx_hash.hex()

    def verify_round(self, round_id: int, expected_hash: str) -> bool:
        """Verify a round's model hash matches the on-chain record."""
        self._connect()
        expected_b32 = self._hex_to_bytes32(expected_hash)
        return self._contract.functions.verifyRound(round_id, expected_b32).call()

    def get_round_details(self, round_id: int) -> dict:
        """Fetch a round's details from the chain."""
        self._connect()
        result = self._contract.functions.getRoundDetails(round_id).call()
        return {
            "round_id": result[0],
            "timestamp": result[1],
            "num_gradients": result[2],
            "model_hash_before": result[3].hex(),
            "model_hash_after": result[4].hex(),
            "num_contributors": len(result[6]),
            "verified": result[8],
        }
