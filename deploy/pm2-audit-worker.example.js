// Example PM2 config: capacity-audit GPU worker (side 3, remote-audit mode).
//
//   pm2 start deploy/pm2-audit-worker.example.js --only audit-worker-3090-0
//
// This runs the timed hot-capacity benchmark on this host's GPU on behalf of a
// (possibly no-GPU) proxy miner. It registers with the worker balancer (Balancer 2);
// the proxy leases it, warms it (hot-start), releases it at B_start, and collects
// the signed proof. This host needs a real GPU + the hot-capacity CUDA wheel.
//
// IMPORTANT:
// - VERATHOS_AUDIT_GPU_CLASS must be the EXACT calibrated class string the proxy
//   advertises in /health (e.g. "NVIDIA GeForce RTX 3090 Ti"), because the balancer
//   routes leases by gpu_class and the audit proves that class.
// - VERATHOS_AUDIT_WORKER_KEY must match the value the balancer hands to proxies
//   (or the value proxies use). The proxy authenticates to this worker with it.
// - Enable remote audit on the PROXY side with:
//     --capacity-audit --capacity-audit-balancer http://<bal>:8081 --capacity-audit-balancer-key <key>
//   (omit --capacity-audit-balancer to keep the audit LOCAL — the default.)

module.exports = {
  apps: [
    {
      name: "audit-worker-3090-0",
      script: ".venv-vllm/bin/python",
      // Use an args ARRAY (not .join) — gpu-class contains spaces.
      args: [
        "-u", "-m", "services.audit_worker",
        "--host", "0.0.0.0",
        "--port", "8095",
        "--worker-id", "audit-3090-0",
        "--public-endpoint", "http://10.0.0.30:8095",
        "--worker-key", "REPLACE_AUDIT_WORKER_KEY",
      ],
      cwd: "/workspace/verathos",
      env: {
        CAPACITY_AUDIT_BALANCER_URL: "http://10.0.0.10:8081",
        CAPACITY_AUDIT_BALANCER_API_KEY: "REPLACE_BALANCER2_KEY",
        VERATHOS_AUDIT_WORKER_KEY: "REPLACE_AUDIT_WORKER_KEY",
        // MUST match this host's real GPU AND the proxy's advertised class.
        VERATHOS_AUDIT_GPU_CLASS: "NVIDIA GeForce RTX 3090 Ti",
      },
      autorestart: true,
      merge_logs: true,
    },
  ],
};
