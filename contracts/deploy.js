/**
 * FzIQ Contract Deployment Script
 * 
 * Usage:
 *   npx hardhat run contracts/deploy.js --network opbnbTestnet
 *   npx hardhat run contracts/deploy.js --network opbnbMainnet
 */

const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log(`Deploying FzIQTrainingLog with account: ${deployer.address}`);
  console.log(`Account balance: ${ethers.formatEther(await deployer.provider.getBalance(deployer.address))} BNB`);

  // The aggregator is the deployer by default.
  // After deployment, call setAggregator() to point to the aggregator service wallet.
  const aggregatorAddress = deployer.address;

  const FzIQTrainingLog = await ethers.getContractFactory("FzIQTrainingLog");
  const contract = await FzIQTrainingLog.deploy(aggregatorAddress);
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log(`FzIQTrainingLog deployed to: ${address}`);
  console.log(`\nAdd to your .env:`);
  console.log(`CONTRACT_ADDRESS=${address}`);

  // Verify deployment
  const currentRound = await contract.getTotalRounds();
  console.log(`\nVerification: currentRound = ${currentRound} (expected 0)`);
  console.log("Deployment complete.");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
