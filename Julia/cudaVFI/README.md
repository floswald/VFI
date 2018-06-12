# CUDA VFI

## Julia

### Proof of concept

this compares GPU vs CPU on a dummy problem that aims to mimic the computations required to perform a dynamic programming solution. This iterates over a 6D array with dimensions

```
da = na
dy = 5
dp = 10
dm = 30
dt = 30
dh = 3
```

where we vary the first dimension, `na`.

```julia
julia> cudaVFI.poc1()
[ Info: running both once to precompile
[ Info: number of elements in V: 270000

[ Info: now timing at na=100:
[ Info: number of elements in V: 13500000
[ Info: cpu = 6.294887158
[ Info: gpu = 0.28379035
[ Info: cpu/gpu = 22.18147008480028

[ Info: now timing at na=150:
[ Info: number of elements in V: 20250000
[ Info: cpu = 14.357773222
[ Info: gpu = 0.6458368
[ Info: cpu/gpu = 22.231272477900326

[ Info: now timing at na=200:
[ Info: number of elements in V: 27000000
[ Info: cpu = 23.078726698
[ Info: gpu = 0.90785486
[ Info: cpu/gpu = 25.421163492549756

[ Info: now timing at na=250:
[ Info: number of elements in V: 33750000
[ Info: cpu = 59.281032424
[ Info: gpu = 1.4818499
[ Info: cpu/gpu = 40.00474816700981

[ Info: now timing at na=300:
[ Info: number of elements in V: 40500000
[ Info: cpu = 79.05968435
[ Info: gpu = 1.8977146
[ Info: cpu/gpu = 41.66047082663815

```