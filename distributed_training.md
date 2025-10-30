    aws s3 sync s3://YOUR_BUCKET/imagenet-1k /opt/dlami/nvme/imagenet-1k --no-progress
    ```

- Recommended NCCL env (single-node):
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=eth0
  ```

- Verify GPUs:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.device_count())"   # should be 4
  ```

#### Single-GPU run
- No code changes to your command style:
  ```bash
  cd /mnt/fb0d7ad2-8ef1-4b6e-b1d2-0c53fa7701b7/TSAI/ERA/session_9/training/ERA4_assignment_9

  # Optional: adjust batch size and epochs in `train_standard.py` if needed
  python train_standard.py
  ```

#### Multi-GPU run (single node, 4 GPUs)
- Use `torchrun` and the new `--distributed` flag:
  ```bash
  cd /mnt/fb0d7ad2-8ef1-4b6e-b1d2-0c53fa7701b7/TSAI/ERA/session_9/training/ERA4_assignment_9

  # Per-GPU batch size is `BATCH_SIZE` in the script. Effective global batch is BATCH_SIZE * 4.
  # Consider reducing LR proportionally if you reduce global batch; or scale LR if you increase it.

  torchrun --standalone --nproc_per_node=4 train_standard.py --distributed
  ```

- Notes:
  - When distributed, the script uses `DistributedSampler`, per-GPU device selection via `LOCAL_RANK`, and `DDP` wrapping.
  - Checkpoints/plots should be written only by rank 0 to avoid races (use the rank-guard as shown above).

### Optional tuning notes
- If `BATCH_SIZE=512` is too large per GPU on A10G 24GB, set `BATCH_SIZE=128` in the script; global becomes 512 with 4 GPUs.
- Scale LR with global batch size if you change it (linear scaling rule is usually fine).
- AMP is already integrated; DDP works well with AMP.

### Switch between single and multi GPU
- Single GPU:
  - Command: `python train_standard.py`
- Multi GPU:
  - Command: `torchrun --standalone --nproc_per_node=4 train_standard.py --distributed`
- No other code changes or flags needed; the new `--distributed` flag cleanly switches behavior.

- If you also want this for `train_hybrid.py` or `train_progessive_resizing.py`, apply the same pattern: add the `--distributed` argument, DDP init, sampler integration, and rank-0 guarded I/O.

- For multi-node later, you’d add `--nnodes`, `--node_rank`, `--rdzv_backend=c10d`, `--rdzv_endpoint=host:port` to the `torchrun` command and keep the same code.

- Quick health checks during training:
  ```bash
  nvidia-smi  # all 4 GPUs should be utilized
  tail -f standard_train/training_log.txt  # rank-0 logs
  ```

- Cleanup:
  - The script calls `dist.destroy_process_group()` when done to release distributed resources.

- S3 checkpoint syncing (optional):
  - Your `s3_utils.py` is already integrated in `train_standard.py`; keep upload calls under a rank-0 check.

- If you’d like, I can also provide a single small helper `ddp_utils.py` to centralize DDP init and rank guards to avoid repeating code across scripts.

- Apply these edits manually or switch to agent mode and I’ll implement them directly for you.

- In short:
  - Edit `data_loader.get_dataloaders` to support DDP samplers.
  - Add `--distributed` flag and DDP init/wrapping in `train_standard.py`.
  - Run with `torchrun` for multi-GPU; run normally for single-GPU.