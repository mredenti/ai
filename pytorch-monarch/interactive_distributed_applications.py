import torch
from monarch.actor import Actor, current_rank, endpoint
from monarch.utils import setup_env_for_distributed
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from dataclasses import dataclass 


from monarch.job import SlurmJob, JobTrait


def create_slurm_job(
    mesh_name: str,
    num_nodes: int,
    gpus_per_node: int,
    time_limit: str = "06:00:00"
) -> SlurmJob:
    """
    Args:
        mesh_name: Name assigned to the primary mesh for this example.
                   A JobTrait can consist of multiple meshes, and
                   Monarch allows for re-attaching to ongoing jobs.
        num_nodes: Number of nodes allocated per mesh
        gpus_per_node: Number of GPUs per node in the mesh

        Note: SlurmJob is just one instance of a Monarch scheduler interface.
              Consult the JobTrait documentation to find one that's right for your usecase.
    """
    default_job_name = "monarch_titan"
    return SlurmJob(
        meshes={mesh_name: num_nodes},
        job_name=default_job_name,
        time_limit=time_limit,
        gpus_per_nodes=gpus_per_node,
        # ... additional args can be passed here
    )

class TrainerActor(Actor):
    """
    Monarch Actor wrapper for TorchTitan's Trainer. 
    
    This actor encapsulates a complete TorchTitan training process, handling 
    initialisation, training loop execution, and cleanup. Each instance runs 
    on a single GPU in the distributed training job. 
    
    The actor's lifetime:
        1. __init__: Initialise with job configuration 
        2. start training: 
            Execute the training loop 
            Destroy process group and release resources 
            
    Attributes:
        job_config: TorchTitan configuration for this trainer 
        uid: Unique identifier for logging (includes rank)
    """
    
    def __init__(self, job_config: "JobConfig") -> None:
        """
        Initialise the trainer actor. 
        
        Args: 
            job_config: TorchTitan JobConfig with training parameters. 
        """
        
        self.job_config = job_config
        
        # current_rank() provides access to this actor's rank in the process mesh 
        self.rank = current_rank().rank 
        self.uid = f"[trainer_{self.rank}]"
        
    @endpoint 
    async def ping_rank(self) -> None:
        """
            A dummy logging function we will use for demonstration purposes.
        """
        logger.info(f"{self.uid} Ping!")    
    
    @endpoint 
    async def start_training(self) -> None:
        """
        Execute the TorchTitan training loop.

        This remote endpoint:
        1. Initializes TorchTitan's logger
        2. Creates a Trainer instance with the job configuration
        3. Runs the training loop
        4. Handles cleanup and error conditions

        The @endpoint decorator makes this method callable from the Monarch
        client, even though it runs on a remote GPU node.

        Raises:
            Exception: Any exception from TorchTitan training is propagated
                      back to the client
        """
        init_logger() 
        trainer: Trainer | None = None
        try:
            # Initilise TorchTitan trainer 
            trainer = Trainer(self.job_config)
            logger.info(f"{self.uid} initialised successfully and starting training loop.")
            
            # Run the training loop 
            trainer.train() 
            
        except Exception as e:
            logger.error(f"{self.uid} training failed with error: {e}")
            if trainer:
                trainer.close()  # Ensure resources are released on error
            # Note error is propagated back to the controller 
            raise e

        else:
            # Training completed successfully, perform cleanup
            trainer.close()  # Ensure resources are released after training
            logger.info(f"{self.uid} training completed successfully.")
            
        finally:
            # clean up distributed process group 
            torch.distributed.destroy_process_group()
            logger.info(f"{self.uid} trainer cleaned up.")
            
        
@dataclass
class RunParams:
    """
    Configuration for cluster resources and training parameters.
    
    Attributes:
        training_steps: Number of training iterations to run
        model_config: Path to TorchTitan model configuration file
        tokenizer: Path to tokenizer directory
        dataset: Dataset to use for training (e.g., 'c4', 'c4_test')
        num_nodes: Number of compute nodes to request
        gpus_per_node: Number of GPUs per node

    Adjust these values based on your model size and available resources.
    """
    
    training_steps: int = 50
    model_config: str = "debug_model.toml"
    tokenizer: str = "tokenizer"
    dataset: str = "c4_test"
    num_nodes: int = 1
    gpus_per_node: int = 0
    

import os
from torchtitan.config import ConfigManager, JobConfig


def make_job_config() -> JobConfig:
    """
    Create a TorchTitan JobConfig from RunParams.

    This function constructs the complete training configuration, including
    parallelism settings, model architecture, and dataset paths
    """
    # Calculate total parallelism based on cluster size
    data_parallel_shard_degree = RunParams.num_nodes * RunParams.gpus_per_node
    output_path = "./outputs"
    # Construct paths relative to script directory
    script_dir = os.getcwd()

    # Build argument list for TorchTitan's ConfigManager
    # These override defaults from the model config file
    default_args = [
        "--job.config_file",
        os.path.join(script_dir, RunParams.model_config),
        "--model.tokenizer_path",
        os.path.join(script_dir, RunParams.tokenizer),
        "--parallelism.data_parallel_shard_degree",
        str(data_parallel_shard_degree),
        "--training.steps",
        str(RunParams.training_steps),
        "--training.dataset",
        RunParams.dataset,
        "--job.dump_folder",
        output_path,
        # continue to configure as needed
    ]
    config_manager = ConfigManager()
    job_config = config_manager.parse_args(default_args)
    return job_config

# Workflow: Reserve Machines → Create Proc Mesh → Configure Logging → Spawn Actors → Train → Cleanup

async def execute_training() -> None:
    """
    Execute the complete distributed training workflow.
    """
    job_config: JobConfig = make_job_config()
    slurm_job: SlurmJob | None = None
    mesh_name = "mesh0"
    
    try:
        # 1. Create a SLURM job with N nodes
        #    This leverages Monarch to reserve a persistent machine allocation
        slurm_job = create_slurm_job(mesh_name, RunParams.num_nodes, RunParams.gpus_per_node)
        job_state = slurm_job.state()

        # 2. Create a process mesh on the machine allocation
        #    This creates one process per GPU across all allocated nodes
        logger.info("Creating process mesh...")
        proc_mesh = job_state.mesh0.spawn_procs({"gpus": RunParams.gpus_per_node})

        # 3. Configure remote logging behavior
        #    - stream_to_client: Forward all remote logs to your local console
        #    - aggregate_window_sec: Batch logs for efficiency
        logger.info("Configuring logging...")
        await proc_mesh.logging_option(
            stream_to_client=True,
            # aggregate_window_sec=None  # Uncomment to disable log batching
        )

        # 4. Setup environment for torch.distributed
        #    This configures torch.distributed across all processes in the mesh
        logger.info("Setting up distributed environment...")
        await setup_env_for_distributed(proc_mesh)

        # 5. Spawn TrainerActor on each GPU
        #    Each process in the mesh creates its own TrainerActor instance
        logger.info("Spawning trainer actors...")
        trainer = proc_mesh.spawn(
            "trainer_actor",  # Name for the actor group
            TrainerActor,  # Actor class to instantiate
            job_config,  # Arguments to __init__
        )

        # 6. Execute the training job across all actors
        #    The .call() method invokes start_training() on all actors in parallel
        logger.info("Starting distributed training...")
        await trainer.start_training.call()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training workflow failed: {e}")

    finally:
        # Always clean up the machine allocation
        if slurm_job:
            await cleanup_job(slurm_job)
            
async def cleanup_job(job: "JobTrait") -> None:
    """
    This function cancels the SLURM job, releasing all reserved nodes back
    to the cluster for other users.

    Args:
        job: A JobTrait, like the one returned from create_slurm_job()

    Note:
        The job will also terminate automatically when the configured TTL
        is exceeded, but explicit cleanup is recommended for long-running
        notebooks or scripts.
    """
    job.kill()
    logger.info("Job terminated successfully")
    
import asyncio


if __name__ == "__main__":
    """
    Run the complete workflow: reserve resources, train, and cleanup.
    """
    logger.info("Starting Monarch + TorchTitan Distributed Training")

    asyncio.run(execute_training())

    logger.info("Workflow completed!")