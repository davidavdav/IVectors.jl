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

Now suppose that you want to use the same data to train an iVector extractor.  The first thing to do, is to extract centralized and scaled statistics for the train data:
```julia
css = map(data->CSstats(ubm, data), x)
```
Again, this may take a while, you might consider `pmap()` instead.  `GaussianMixtures::CSstats` stores the zeroth and first order statistics w.r.t. the UBM.  The type is parameterized, and it apprears that `map()` is better at registering the type than a comprehension would do, so we advice to use `map()` here.

An iVector extractor can be trained now for a given number of voices (target dimension):
```julia
using IVectors
nvoices = 100
ie = IExtractor(css, nvoices; nIter=7)
```
### Saving and loading the IVector extractor

Saving and loading the IVector extractor relies on the `FileIO` and `JLD` packages.
```julia
using FileIO
save("file.iex", ie)
save("iex-noext" ie) ## another copy with a funny name
```
`FileIO` can recognize the `format"IExtractor"` format by the `.iex` extension.  The registration of the format happens in `IVectors`, not in `FileIO`.
```julia
using IVectors
ie = load("file.iex") ## by extension
ie = load(File(format"IExtractor", "iex-noext")) ## by format
ie = load("iex-noext", IExtractor) ## by type
```
### Data representation
Although in literature the IVector extractor is always portrayed as a single "tall" matrix of `CF x nvoices`, where `C` is the number of Gaussian components and `F` feature dimension, we internally represent `T` as a length-`C` vector of `F x nvoices` matrices.  This is also how it is serialized to disc.  Further, upon loading or calculation of the `IExtractor`, a length-`C` vector of `nvoices x nvoices` matrices is kept in memory for computational efficiency. 

## Extracting iVectors
iVectors are extracted using the same `CSstats` structure.  For a data matrix `data`:
```julia
ivec = ivector(ie, CSstats(ubm, data))
```
A slightly more efficient way to extract ivectors is to use a bunch of files simultaneously:
```julia
blas_set_num_threads(2)
css = map(data->CSstats(ubm, data), x) ## x is vector of data matrices
ivecs = ivector(ie, css) ## a Vector of Vectors
```
This implementation uses `Base.BLAS.gemm!()` to compute ivectors simultaneously for multiple `CSstats` objects, we have found that we're not gaining much speed by setting the number of openblas threads too high.

## Status

Current status of the package is that it
 - needs more validation
 - needs support for full covariance UBMs.  This should happen in `CSstats`, though.
 - scoring (cosine distance is trivial)
 - PLDA scoring
