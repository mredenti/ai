 A few comments on the implementation:

- no_sync() for grad accumulation: without this, every micro-step triggers an AllReduce,
        wasting bandwidth on intermediate gradients that will just be overwritten. This is 
    the DDP equivalent of what Trainer/Accelerate do automatically. 
- Mixed precision via torch.autocast: replaces the bf16=True flag. For pure bf16 you do not need 
    a GradScaler, unlike fp16
- Saving with model.module: DDP wraps your model, so .module gets the underlying nn.Module back. Only rank 0
writes to disk to avoid races. 
- Bucketing: Increase it if you have a fast interconnect and want fewer, larger AllReduces. 
