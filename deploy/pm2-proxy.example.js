// Example PM2 config: proxy miner on a GPU host (single process).
//
//   pm2 start deploy/pm2-proxy.example.js --only proxy-miner-a
//
// This one process:
//   - forwards /chat + /inference to the remote LLM pool via Balancer 1
//     (no local vLLM),
//   - answers /identity + /health locally (EVM key + advertised GPU), and
//   - runs the capacity audit LOCALLY on THIS host's GPU (no worker balancer).
//
// IMPORTANT:
// - This host needs a real GPU + the hot-capacity CUDA wheel installed: the
//   capacity audit runs bench_combined.py on --gpu-index 0 here.
// - The advertised GPU MUST equal this host's real local GPU — the audit is
//   validated against the advertised class's calibration.
// - --model-id / --quant must pass the validator capacity model gate for the
//   advertised VRAM.
// - Set VERATHOS_ADVERTISED_GPU_NAME via env when the name contains spaces.
// - --proxy-balancer points at Balancer 1. The monitor mounts /pick under /api,
//   so include the /api suffix (e.g. http://10.0.0.10:3840/api). If you run a
//   balancer that exposes /pick at the root, drop the suffix.
// - Capacity audit requires hosted subnet runtime config (windows_per_epoch=5
//   on mainnet). On startup you should see:
//     "Applied runtime subnet config version=... source=server"
//   If fetch fails, seed the cache once:
//     mkdir -p ~/.verathos && curl -o ~/.verathos/subnet_config_cache.json https://api.verathos.ai/v1/subnet-config

module.exports = {
  apps: [
    {
      name: "proxy-miner-a",
      script: ".venv-vllm/bin/python",
      args: [
        "-u", "-m", "neurons.miner",
        "--wallet", "verathos",
        "--hotkey", "miner1",
        "--netuid", "96",
        "--subtensor-network", "finney",
        "--model-id", "Qwen/Qwen3.5-9B",
        "--quant", "fp16",
        "--endpoint", "https://proxy.example.com:31123",
        "--capacity-audit",
        "--proxy-mode",
        "--proxy-balancer", "http://10.0.0.10:3840/api",
        "--proxy-balancer-key", "REPLACE_BALANCER1_KEY",
        "--proxy-llm-key", "REPLACE_PROXY_LLM_KEY",
        "--advertised-vram-gb", "80",
        "--advertised-gpu-uuids", "REPLACE_LOCAL_GPU_UUID",
        "--",
        "--port", "9062",
        "--proxy-mode",
        "--skip-gpu-check",
      ],
      cwd: "/workspace/verathos",
      env: {
        // MUST match this host's real local GPU (the one that runs the audit).
        VERATHOS_ADVERTISED_GPU_NAME: "NVIDIA A100-SXM4-80GB",
        VERATHOS_ADVERTISED_VRAM_GB: "80",
        // Set to 0 if Balancer 1 returns https:// GPU endpoints with self-signed certs.
        PROXY_UPSTREAM_VERIFY_SSL: "0",
      },
      autorestart: true,
      merge_logs: true,
    },
  ],
};
