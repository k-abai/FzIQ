// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FzIQTrainingLog
 * @notice Immutable audit log for FzIQ federated training rounds.
 * @dev Deployed on opBNB. Each round writes: model hash before/after,
 *      scenario hashes, agent IDs, and combined scores.
 *      Any researcher can verify the full training trajectory onchain.
 *
 * @author Keke Abai — Boston University, 2026
 */
contract FzIQTrainingLog {

    // ─────────────────────────────────────────────────────────── Data types ──

    struct TrainingRound {
        uint256 roundId;
        uint256 timestamp;
        uint256 numGradients;
        bytes32 modelHashBefore;
        bytes32 modelHashAfter;
        bytes32[] scenarioHashes;
        address[] agentIds;
        uint256[] scores;        // scaled by 1e4 for integer storage (0–10000)
        bool verified;
    }

    struct AgentRecord {
        address agentId;
        uint256 totalContributions;
        uint256 cumulativeScore;     // sum of all scores * 1e4
        uint256 reliabilityScore;    // running average * 1e4 (= cumulativeScore / totalContributions)
    }

    // ───────────────────────────────────────────────────────────── Storage ──

    mapping(uint256 => TrainingRound) public rounds;
    mapping(address => AgentRecord)   public agents;

    uint256 public currentRound;
    address public aggregator;     // only this address may write round logs
    address public owner;

    // ────────────────────────────────────────────────────────────── Events ──

    event RoundLogged(
        uint256 indexed roundId,
        bytes32 modelHashBefore,
        bytes32 modelHashAfter,
        uint256 numContributors
    );

    event AgentContributed(
        address indexed agentId,
        uint256 indexed roundId,
        uint256 score
    );

    event AggregatorChanged(address oldAggregator, address newAggregator);

    // ─────────────────────────────────────────────────────────── Modifiers ──

    modifier onlyAggregator() {
        require(msg.sender == aggregator, "FzIQ: caller is not the aggregator");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "FzIQ: caller is not the owner");
        _;
    }

    // ──────────────────────────────────────────────────────────── Constructor

    constructor(address _aggregator) {
        aggregator = _aggregator;
        owner = msg.sender;
        currentRound = 0;
    }

    // ──────────────────────────────────────────────────── Write functions ──

    /**
     * @notice Log a completed training round.
     * @param modelHashBefore  SHA-256 of metamodel params before this aggregation
     * @param modelHashAfter   SHA-256 of metamodel params after this aggregation
     * @param scenarioHashes   SHA-256 of each training scenario (one per gradient)
     * @param agentIds         Ethereum address of each contributing agent
     * @param scores           Combined human+CNN score per gradient, scaled ×1e4
     */
    function logRound(
        bytes32 modelHashBefore,
        bytes32 modelHashAfter,
        bytes32[] calldata scenarioHashes,
        address[] calldata agentIds,
        uint256[] calldata scores
    ) external onlyAggregator {
        require(agentIds.length > 0,                         "FzIQ: no contributors");
        require(agentIds.length == scores.length,            "FzIQ: agentIds/scores length mismatch");
        require(agentIds.length == scenarioHashes.length,    "FzIQ: agentIds/hashes length mismatch");

        TrainingRound storage round = rounds[currentRound];
        round.roundId        = currentRound;
        round.timestamp      = block.timestamp;
        round.numGradients   = agentIds.length;
        round.modelHashBefore = modelHashBefore;
        round.modelHashAfter  = modelHashAfter;
        round.verified       = true;

        for (uint256 i = 0; i < agentIds.length; i++) {
            round.scenarioHashes.push(scenarioHashes[i]);
            round.agentIds.push(agentIds[i]);
            round.scores.push(scores[i]);

            AgentRecord storage agent = agents[agentIds[i]];
            agent.agentId = agentIds[i];
            agent.totalContributions++;
            agent.cumulativeScore   += scores[i];
            agent.reliabilityScore   = agent.cumulativeScore / agent.totalContributions;

            emit AgentContributed(agentIds[i], currentRound, scores[i]);
        }

        emit RoundLogged(currentRound, modelHashBefore, modelHashAfter, agentIds.length);
        currentRound++;
    }

    // ──────────────────────────────────────────────────── Read functions ──

    /**
     * @notice Verify that a round's post-aggregation model hash matches a given value.
     * @return true if the stored hash equals expectedHash
     */
    function verifyRound(uint256 roundId, bytes32 expectedHash) external view returns (bool) {
        return rounds[roundId].modelHashAfter == expectedHash;
    }

    /**
     * @notice Retrieve the agent reliability score (0–10000 = 0.00–1.00).
     */
    function getAgentReliability(address agentId) external view returns (uint256) {
        return agents[agentId].reliabilityScore;
    }

    /**
     * @notice Get full round details for audit / reproduction.
     */
    function getRoundDetails(uint256 roundId) external view returns (TrainingRound memory) {
        return rounds[roundId];
    }

    /**
     * @notice Get total number of completed rounds.
     */
    function getTotalRounds() external view returns (uint256) {
        return currentRound;
    }

    // ─────────────────────────────────────────────────────────── Admin ──

    /**
     * @notice Transfer aggregator rights to a new address.
     */
    function setAggregator(address newAggregator) external onlyOwner {
        require(newAggregator != address(0), "FzIQ: zero address");
        emit AggregatorChanged(aggregator, newAggregator);
        aggregator = newAggregator;
    }
}
