class: middle, center, title-slide
count: false

# Distributed statistical inference
# with pyhf enabled through funcX

.huge.blue[Matthew Feickert]<br>
.huge[(University of Illinois at Urbana-Champaign)]
<br><br>
[matthew.feickert@cern.ch](mailto:matthew.feickert@cern.ch)

[vCHEP 2021](https://indico.cern.ch/event/948465/)

May 20th, 2021

.middle-logo[]

---
# Notes to self and reviewers

<br><br>

- Parallel session talk on .bold[Thursday, 2021-05-20]
- Talk starts at .bold[15:13] CERN time (08:13 Illinois time)
- Talk time: .bold[8 minutes] + 5 minutes for questions
   - Need to focus on economy of time

---
# Authors

<br><br>

.grid[
.kol-1-4.center[
.circle.width-80[![Lukas](figures/collaborators/heinrich.jpg)]

[Lukas Heinrich](https://github.com/lukasheinrich)

CERN
]
.kol-1-4.center[
.circle.width-80[![Matthew](https://avatars2.githubusercontent.com/u/5142394)]

[Matthew Feickert](https://www.matthewfeickert.com/)

Illinois
]
.kol-1-4.center[
.circle.width-80[![Giordon](figures/collaborators/stark.jpg)]

[Giordon Stark](https://github.com/kratsg)

UCSC SCIPP
]
.kol-1-4.center[
.circle.width-75[![Ben](https://avatars2.githubusercontent.com/u/8229875)]

[Ben Galewsky](https://bengalewsky.github.io/about/)

NCSA/Illinois
]
]

---
# Fitting as a Service with `pyhf` on HPCs

.kol-1-2[
- HPC facilities provide an opportunity to efficiently perform the statistical inference of LHC data
- Can pose problems with orchestration and efficient scheduling
- Want to leverage pyhf hardware accelerated backends at HPC sites for real analysis speedup
   - Reduce fitting time from hours to minutes
- Deploy a _(fitting) Function as a Service_ (FaaS) powered through funcX
- Example use cases:
   - Large scale ensemble fits for statistical combinations
   - Large dimensional scans of theory parameter space (e.g. pMSSM scans)
   - Pseudo-experiment generation ("toys")
]
.kol-1-2[
 .center.width-100[![carbon_pyhf_HVTWZ_3500_fit](figures/carbon_pyhf_HVTWZ_3500_fit.png)]
 ATLAS workspace that takes over an hour on ROOT fit in under 2 minutes with pyhf on GPU
]

---
# Fitting as a Service Methods and Technologies

.kol-1-2[
.center.width-50[[![pyhf-logo](https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/pyhf-logo-small.png)](https://pyhf.readthedocs.io/)]
- Pure Python implementation of the `HistFactory` statistical specification for multi-bin histogram-based analysis
- Supports multiple computational backends and optimizers (defaults of NumPy and SciPy)
- JAX, TensorFlow, and PyTorch backends can leverage _hardware acceleration_ (GPUs, TPUs) and _automatic differentiation_
- Can outperform C++ implementations of `HistFactory`
]
.kol-1-2[
.center.width-80[[![funcX-light](figures/funcX-light.png)](https://funcx.readthedocs.io/en/latest/)]
- High-performance FaaS platform
- Designed to orchestrate _scientific workloads_ across _heterogeneous computing resources_ (clusters, clouds, and supercomputers) and task execution providers (HTCondor, Slurm, Torque, and Kubernetes)
- Leverages [Parsl](https://parsl.readthedocs.io/) for efficient parallelism and managing concurrent task execution
- Allows users to register and then execute Python functions in "serverless supercomputing" workflow
]

---
# funcX Endpoints on HPC

.kol-1-2[
- funcX endpoint: logical entity that represents a compute resource
]
.kol-1-2[
- .bold[Would be great to have a figure go here]
]

---
# Scheduling with funcX

.kol-2-3[
```python
import json
from pathlib import Path
from time import sleep

from funcx.sdk.client import FuncXClient
from pyhf.contrib.utils import download


def prepare_workspace(data):
    import pyhf

    return pyhf.Workspace(data)

if __name__ == "__main__":
    fxc = FuncXClient()
    # Register function and execute on worker node
    prepare_func = fxc.register_function(prepare_workspace):w
    prepare_task = fxc.run(
        bkgonly_workspace, endpoint_id=pyhf_endpoint, function_id=prepare_func
    )
```
]
.kol-1-3[
- Points walking through code
- Need to also fixup the code to make it fit
]

---
# Scaling of Statistical Inference

- RIVER
- NCSA Bluewaters (CPU)
- XSEDE Expanse (GPU JAX)

---
# Performance

- Add results table

---
# Summary

- Demonstrated the ability to parallelize and accelerate statistical inference of physics analyses on
HPC systems through a FaaS solution

---
class: end-slide, center

.large[Backup]

---
# Why use funcX as opposed to Dask?

- funcX provides a _managed service_ secured by Globus Auth

- Endpoints can be set up by a site administrator and shared with authorized users through Globus Auth Groups

- [Testing has shown](https://parsl.readthedocs.io/en/stable/userguide/performance.html) that Dask struggles to .bold[scale up to thousands of nodes], whereas the funcX High Throughput Executor (HTEX) provided through [Parsl](https://parsl.readthedocs.io/) scales efficiently

[.center.width-80[![Scaling Comparison](https://parsl.readthedocs.io/en/stable/_images/strong-scaling.png)]](https://parsl.readthedocs.io/en/stable/userguide/performance.html)

---
# References

1. Lukas Heinrich, .italic[[Distributed Gradients for Differentiable Analysis](https://indico.cern.ch/event/960587/contributions/4070325/)], [Future Analysis Systems and Facilities Workshop](https://indico.cern.ch/event/960587/), 2020.
2. Babuji, Y., Woodard, A., Li, Z., Katz, D. S., Clifford, B., Kumar, R., Lacinski, L., Chard, R., Wozniak, J., Foster, I., Wilde, M., and Chard, K., Parsl: Pervasive Parallel Programming in Python. 28th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC). 2019. https://doi.org/10.1145/3307681.3325400

---

class: end-slide, center
count: false

The end.
