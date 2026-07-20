// PM2 ecosystem config for the `advertised-hardware` branch.
//
// This is the ORIGINAL GPU miner (local inference + local capacity audit),
// with one addition: it can DECLARE the hardware it reports via /health —
// GPU class name, VRAM, GPU count, compute capability, GPU UUIDs — overriding
// auto-detection. Validators read /health to schedule capacity audits, so the
// class you advertise is the class you are audited as.
//
//   cp ecosystem.config.advertised-hardware.example.js ecosystem.config.js
//   pm2 start ecosystem.config.js --only miner
//
// After changing args, delete + recreate the process (plain restart won't
// reload argv/env):
//   pm2 delete miner && pm2 start ecosystem.config.js --only miner
//   # env-only changes:  pm2 restart miner --update-env
//
// ── The advertised-hardware flags (all optional) ──────────────────────────
//   --advertised-gpu-name  "<exact calibrated class string>"   e.g. "NVIDIA A40"
//   --advertised-vram-gb   <int>        snapped to nearest marketed size (46 -> 48)
//   --advertised-gpu-count <int>        default 1
//   --advertised-compute-capability <x.y>   e.g. 8.0
//   --advertised-gpu-uuids "<uuid1,uuid2>"
//
// The override takes effect ONLY when BOTH --advertised-gpu-name AND
// --advertised-vram-gb are set. It changes what /health, /identity and the
// capacity-audit class report — the PHYSICAL GPU still governs which model
// loads. Equivalent env vars work too (see the `env` block below):
//   VERATHOS_ADVERTISED_GPU_NAME / _VRAM_GB / _GPU_COUNT /
//   _COMPUTE_CAPABILITY / _GPU_UUIDS
//
// ⚠️  IMPORTANT — do not advertise UP. The capacity audit runs LOCALLY on your
//     real GPU and must finish the advertised class's workload within that
//     class's deadline. Advertise a class your physical card can actually pass
//     (same as, or lower than, your real GPU). Advertising a faster class than
//     you own makes the local audit miss the deadline and fail. There is no
//     penalty for being faster than the advertised class.

module.exports = {
  apps: [
    // ── Miner (advertised hardware) ───────────────────────────────
    // Required: --wallet, --hotkey, --netuid, --endpoint
    // Model:    --model-id auto  (or --model-id <id> --quant <quant>)
    // Capacity: --capacity-audit enables the local capacity-audit worker.
    {
      name: "miner",
      script: ".venv-vllm/bin/python",
      args: [
        "-u -m neurons.miner",
        "--wallet <WALLET> --hotkey <HOTKEY>",
        "--netuid 96 --subtensor-network finney",
        "--endpoint https://<YOUR_PUBLIC_IP_OR_DOMAIN>:<PORT>",
        "--model-id auto",
        "--capacity-audit",
        // ── advertised hardware (edit or delete) ──
        '--advertised-gpu-name "NVIDIA A40"',
        "--advertised-vram-gb 48",
        // "--advertised-gpu-count 1",
        // "--advertised-compute-capability 8.6",
        // '--advertised-gpu-uuids "GPU-xxxxxxxx-....,GPU-yyyyyyyy-...."',
        // ── server args after `--` (port, explicit model/quant, etc.) ──
        "-- --port 11005",
      ].join(" "),
      cwd: "<REPO_ROOT>",
      env: {
        // Alternative to the CLI flags above — uncomment to use env instead.
        // VERATHOS_ADVERTISED_GPU_NAME: "NVIDIA A40",
        // VERATHOS_ADVERTISED_VRAM_GB: "48",
        // VERATHOS_ADVERTISED_GPU_COUNT: "1",
        // VERATHOS_ADVERTISED_COMPUTE_CAPABILITY: "8.6",
        // VERATHOS_ADVERTISED_GPU_UUIDS: "GPU-xxxx...,GPU-yyyy...",
      },
      // GPU-bound — do NOT auto-restart. Crash loops waste VRAM.
      autorestart: false,
      max_restarts: 0,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },
  ],
};

// ── Example B: advertise an RTX 3090 (24 GB, class "NVIDIA GeForce RTX 3090") ──
//   '--advertised-gpu-name "NVIDIA GeForce RTX 3090"',
//   "--advertised-vram-gb 24",
//
// Calibrated class strings must match the capacity table exactly, e.g.:
//   "NVIDIA GeForce RTX 3090"      (24 GB)
//   "NVIDIA GeForce RTX 3090 Ti"   (24 GB)
//   "NVIDIA GeForce RTX 4090"      (24 GB)
//   "NVIDIA GeForce RTX 5090"      (32 GB)
//   "NVIDIA A40"                   (48 GB, reported as 46 -> snapped to 48)
//   "NVIDIA L40S"                  (48 GB)
//   "NVIDIA A100-SXM4-40GB"        (40 GB)
//   "NVIDIA A100 80GB PCIe"        (80 GB)
//   "NVIDIA A100-SXM4-80GB"        (80 GB)
