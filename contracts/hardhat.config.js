require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },
  networks: {
    hardhat: {},
    opbnbTestnet: {
      url: process.env.OPBNB_RPC_URL || "https://opbnb-testnet-rpc.bnbchain.org",
      chainId: 5611,
      accounts: process.env.AGGREGATOR_PRIVATE_KEY
        ? [process.env.AGGREGATOR_PRIVATE_KEY]
        : [],
      gasPrice: 1000000000, // 1 gwei
    },
    opbnbMainnet: {
      url: "https://opbnb-mainnet-rpc.bnbchain.org",
      chainId: 204,
      accounts: process.env.AGGREGATOR_PRIVATE_KEY
        ? [process.env.AGGREGATOR_PRIVATE_KEY]
        : [],
      gasPrice: 1000000000,
    },
  },
  etherscan: {
    apiKey: {
      opbnbTestnet: process.env.BSCSCAN_API_KEY || "",
    },
    customChains: [
      {
        network: "opbnbTestnet",
        chainId: 5611,
        urls: {
          apiURL: "https://api-opbnb-testnet.bscscan.com/api",
          browserURL: "https://opbnb-testnet.bscscan.com",
        },
      },
    ],
  },
};
