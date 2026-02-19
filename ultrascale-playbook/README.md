# Training ultrascale LLMs

**How to scale the training of large language models (LLMs) from one GPU to tens, hundreds, and even thousands of GPUs?**

As the scale of the data and compute used to train these models has grown, various techniques have been adopted...

- data parallelism 
- tensor parallelism 
- pipeline parallelism
- context parallelism 
- ZeRO and Kernel Fusion 

... to make sure GPUs are highly utilised at all times. This significantly reduces training time and makes the most efficient use of this expensive hardware. 

*Note* It would be good to give a motivation by showcasing the first bottleneck (memory?) that may arise when increasing the data availability?

Use cases for distribute training:

- training initial LLMs
- fine-tuning large models on specialised data 

In these notebooks we will go through these techniques, from the simplest to the most refined ones.

## Practical Considerations 

In practice, how to *actually* scale your LLM training depends on your infrastructure, such as the kind of chips used, interconnect, etc. so there is not single recipe for this. This is where benchmarking comes in. 

There are three key challenges:

1. **Memory Usage**: this is a hard limitation, if a training step doesn't fit in memory, training can not proceed
2. **Compute Efficiency** we want our hardware to spend most of the time computing, so we need to reduce time spent on data transfers or waiting for other GPUs to perform work
3. **Communication Overhead** we want to minimise communication overhead, as it keeps GPUs idle. Make best use of intra-node (fast) and inter-node (slower) bandwidths and to overlap communication with compute as mush as possible. 

## Model Hyperparameters 

- **batch size (bs)**: if affects both model convergence and throughput (?)

## Fun Facts 

The batch size and the training corpus size have been steadily increasing over the years: Llama 1 was trained with a batch size of ~4M tokens for 1.4 trillion tokens, while DeepSeek was trained with a batch size of ~60M tokens for 14 trillion tokens.

- We run into our first challenge when scaling the training of our model to these large batch sizes: out-of-memory (OOM) issues. What should we do when our GPU doesn’t have enough memory to hold a full batch of our target batch size? Can we not do lazy loading of the inputs in the batch??