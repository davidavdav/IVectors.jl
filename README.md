IVectors.jl
=======

i-vector training, extraction and scoring routines

This is a small package that does basic i-vector training and extraction, a framework that is used in Automatic Speaker Recognition.  The package relies on [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl) for training a Universal Background Model (UBM) and computing statistics.  

## Install

```julia
Pkg.clone("https://github.com/davidavdav/IVectors.jl")
```

## Training an ivector extractor. 

Data is represented as a matrix with data "running down", i.e., the data matrix is formed by a stack of features as "row vectors".  Typically, a data matrix represents a single audio file, and is variable in the size of the first dimension, but fixed in the second, which is the number of features for each data point. 

Suppose you have a vector of such data matrices, `x::Vector{Matrix}`, then you can first train a diagonal covariance UBM with `GaussianMixtures`, using

```julia
using GaussianMixtures
ngauss = 1024
ubm = GMM(ngauss, Data(x), kind=:diag)
```
This may take a while, check out parallelization options in [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl).  

Now suppose that you want to use the same data to train an iVector extractor.  The first thing to do, is to extract centralized (but not scaled) statistics for the train data:
```julia
cs = map(data->Cstats(ubm, data), x)
```
Again, this may take a while, you might consider `pmap()` instead.  `GaussianMixtures::Cstats` stores the zeroth and first order statistics w.r.t. the UBM.  The type is parameterized, and it apprears that `map()` is better at registering the type than a comprehension would do, so we advice to use `map()` here.

An iVector extractor can be trained now for a given number of voices (target dimension):
```julia
using IVectors
nvoices = 100
ie = IExtractor(ubm, cs, nvoices)
```

## extracting iVectors
iVectors are extracted using the same `Cstats` structure.  For a data matrix `data`:
```julia
ivec = ivector(ie, Cstats(ubm, data))
```

## Status

Current status of the package is that it 
 - needs validation
 - needs parallelization optimization
 - needs change from `Cstats` to `CSstats` (centered and scaled stats)
 - needs support for full covariance UBMs. 

