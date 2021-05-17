from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor

from parsl.providers import LocalProvider

...

config = Config(
    executors=[
        HighThroughputExecutor(
            label="fe.cs.uchicago",
            address=address_by_hostname(),
            provider=SlurmProvider(
                channel=LocalChannel(),
                nodes_per_block=NODES_PER_JOB,
                init_blocks=1,
                partition="general",
                launcher=SrunLauncher(
                    overrides=(
                        f"hostname; srun --ntasks={TOTAL_WORKERS} "
                        f"--ntasks-per-node={WORKERS_PER_NODE} "
                        f"--gpus-per-task=rtx2080ti:{GPUS_PER_WORKER} "
                        f"--gpu-bind=map_gpu:{GPU_MAP}"
                    )
                ),
                walltime="01:00:00",
            ),
        )
    ],
)
