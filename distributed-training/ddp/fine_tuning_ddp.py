import os
import math
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk

WARMUP_RATIO  = 0.03
LOGGING_STEPS = 25


## DDP setup -----------------------------
def ddp_setup():
    """
    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    Works identically for single-node and multi-node launches.
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


## Trainer class -----------------------------
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        eval_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        save_every: int,
        snapshot_path: str,
        grad_accum_steps: int = 1,
    ) -> None:
        self.local_rank  = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.eval_data  = eval_data
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.grad_accum_steps = grad_accum_steps
        self.epochs_run  = 0
        self.global_step = 0

        if os.path.exists(snapshot_path):
            print(f"[GPU{self.global_rank}] Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=False,
            bucket_cap_mb=25,
        )

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run  = snapshot["EPOCHS_RUN"]
        self.global_step = snapshot.get("GLOBAL_STEP", 0)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch, micro_step):
        # Skip gradient AllReduce on accumulation steps — only sync on the
        # step that will actually call optimizer.step().
        is_accum = (micro_step + 1) % self.grad_accum_steps != 0
        sync_ctx = self.model.no_sync() if is_accum else torch.enable_grad()

        with sync_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss / self.grad_accum_steps
            loss.backward()

        if not is_accum:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % LOGGING_STEPS == 0:
                # Reduce loss across ranks so the log line shows the true average.
                loss_t = torch.tensor(loss.item() * self.grad_accum_steps, device=self.local_rank)
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                if self.global_rank == 0:
                    print(
                        f"step {self.global_step:>6} | loss {loss_t.item():.4f} "
                        f"| lr {self.scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

    def _run_eval(self):
        """All ranks participate (needed for dist.all_reduce); only rank 0 prints."""
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in self.eval_data:
                batch = {k: v.to(self.local_rank, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                n += 1
        avg = torch.tensor(total_loss / max(n, 1), device=self.local_rank)
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        self.model.train()
        return avg.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        if self.global_rank == 0:
            print(f"[GPU0] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}", flush=True)
        self.train_data.sampler.set_epoch(epoch)   # reshuffle each epoch
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        for micro_step, batch in enumerate(self.train_data):
            batch = {k: v.to(self.local_rank, non_blocking=True) for k, v in batch.items()}
            self._run_batch(batch, micro_step)

        # Eval at end of epoch (all ranks run, rank 0 prints)
        eval_loss = self._run_eval()
        if self.global_rank == 0:
            print(
                f"[GPU0] Epoch {epoch} | eval loss {eval_loss:.4f} "
                f"| elapsed {time.time()-t0:.1f}s",
                flush=True,
            )

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN":  epoch,
            "GLOBAL_STEP": self.global_step,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


## Factory: load model, data, optimizer -----------------------------
def load_train_objs():
    dataset_path = "/leonardo_work/tra26_minwinsc/DATA/Bitext-customer-support-llm-chatbot-training-dataset"
    model_id     = "/leonardo_work/tra26_minwinsc/models/Llama-3.2-1B-Instruct"
    max_seq_len  = 1024

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset: load → format with chat template → tokenize
    raw = load_from_disk(dataset_path)

    def format_example(example):
        messages = [
            {"role": "system",    "content": "You are a helpful and courteous customer support assistant."},
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return example

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,   # collator pads to longest in batch
        )

    formatted  = raw.map(format_example, remove_columns=raw.column_names)
    tokenized  = formatted.map(tokenize, batched=True, remove_columns=["text"])
    split      = tokenized.train_test_split(test_size=0.1, seed=42)
    train_set  = split["train"]
    eval_set   = split["test"]

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Collator (handles dynamic padding + builds `labels` for causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return train_set, eval_set, model, optimizer, collator


## DataLoader factory -----------------------------
def prepare_dataloader(dataset, batch_size: int, collator, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,   # DistributedSampler handles ordering
        sampler=DistributedSampler(dataset, shuffle=shuffle),
        collate_fn=collator,
        num_workers=2,
    )


## Main -----------------------------
def main(save_every: int, total_epochs: int, batch_size: int, grad_accum_steps: int,
         snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_set, eval_set, model, optimizer, collator = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size, collator, shuffle=True)
    eval_data  = prepare_dataloader(eval_set,  batch_size, collator, shuffle=False)

    # Cosine LR schedule with linear warmup
    steps_per_epoch = math.ceil(len(train_data) / grad_accum_steps)
    total_opt_steps = steps_per_epoch * total_epochs
    warmup_steps    = int(WARMUP_RATIO * total_opt_steps)

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

    trainer = Trainer(
        model, train_data, eval_data, optimizer, scheduler,
        save_every, snapshot_path, grad_accum_steps,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Llama full fine-tuning with DDP')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every',   type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size',       default=2, type=int, help='Per-GPU batch size (default: 2)')
    parser.add_argument('--grad_accum_steps', default=4, type=int, help='Gradient accumulation steps (default: 4)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.grad_accum_steps)
