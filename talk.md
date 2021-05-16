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
# Problem statement

- HPC facilities provide an opportunity to efficiently perform the statistical inference of LHC data
- Can pose problems with orchestration and efficient scheduling
- We can present a framework
- ...
- Throw in physics motivation for pMSSM scans and pseudo-experiments

---
# Fitting as a Service Methods and Technologies

.kol-1-2[
.bold[pyhf]
]
.kol-1-2[
.bold[funcX]
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

- Testing has shown that Dask struggles to .bold[scale up to thousands of nodes], whereas the funcX High Throughput Executor (HTEX) scales efficiently

.center.width-80[![Scaling Comparison](https://parsl.readthedocs.io/en/stable/_images/strong-scaling.png)]

---
# References

1. Lukas Heinrich, .italic[[Distributed Gradients for Differentiable Analysis](https://indico.cern.ch/event/960587/contributions/4070325/)], [Future Analysis Systems and Facilities Workshop](https://indico.cern.ch/event/960587/), 2020.

---

class: end-slide, center
count: false

The end.
