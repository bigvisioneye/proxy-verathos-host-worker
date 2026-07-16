// Example PM2 config: inference GPU (LLM pool) — vLLM + proofs, accepts
// proxy-forwarded traffic. Register this endpoint in Balancer 1 (verathos-monitor)
// so the proxy's /pick can route to it.
//
//   pm2 start deploy/pm2-inference-gpu.example.js --only inference-qwen9b
//
// VERATHOS_PROXY_LLM_KEY must match the proxy miner's --proxy-llm-key so only
// the proxy can reach /chat + /inference.

module.exports = {
  apps: [
    {
      name: "inference-qwen9b",
      script: ".venv-vllm/bin/python",
      args: [
        "-u", "-m", "verallm.api.server",
        "--model-id", "Qwen/Qwen3.5-9B",
        "--quant", "fp16",
        "--port", "8000",
        "--host", "0.0.0.0",
      ].join(" "),
      cwd: "/workspace/verathos",
      env: {
        VERATHOS_PROXY_LLM_KEY: "REPLACE_PROXY_LLM_KEY",
      },
      autorestart: true,
      merge_logs: true,
    },
  ],
};
