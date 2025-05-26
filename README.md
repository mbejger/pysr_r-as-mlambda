# pySR script for the ''Approximating neutron-star radii using gravitational-wave only measurements with symbolic regression'' paper 
# (https://arxiv.org/abs/2504.19962)

The script reads tabulated data (`train.data` and `test.data` - 4 columns: `R`, `M`, `Lambda`, `k2`) and performs symbolic regression using the `pysr` library.

```
python3 pysr_r_as_mlambda.py train.data test.data
```

The output model is written to `outputs' directory. It contains the best symbolic regression model found during the optimization process amd described in `arXiv:2504.19962`.  
