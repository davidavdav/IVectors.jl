## ivector.jl  Various routines for ivector extraction
## (c) 2013--2014 David A. van Leeuwen

## a global for timing alternative code...
alt=false

function setalt(a)
    global alt=a
end

## Kenny-order supervectors correspond to GMM-order matrices by stacking row-wise, i.e., svec(X) = vec(X')
## for g in Gaussians for f in features X[g,f] end end
svec(x::Matrix) = vec(x')

## I + x, but doing I + x in-memory (not used)
function Iplusx!{T}(x::Matrix{T})
    for i in 1:size(x,1)
        x[i,i] += one(T)
    end
    return x
end

## This constructs the T matrix from the IExtractor vector-of-component-T's (not used)
## This is actually inefficient, because it declares a lot of memory
Tᵀ(ie::IExtractor) = hcat([Tc' for Tc in ie.T]...)

## Some utlity functions
nfea(ie::IExtractor) = size(first(ie.T), 1)
nvoices(ie::IExtractor) = size(first(ie.T), 2)
ngauss(ie::IExtractor) = length(ie.T)
Base.size(ie::IExtractor) = length(ie.T), size(first(ie.T))...

## compute variance and mean of the posterior distribution of the hidden variables
## s: centered and scaled statistics (.n .f),
## ie.T: ng x (nfea x nvoices) matrix
## result is a Nvoices vector and Nvoices x Nvoices matrix
function posterior{Float<:AbstractFloat}(ie::IExtractor{Float}, s::CSstats{Float}; Linv=true)
    nv = nvoices(ie)
    cov = eye(Float, nv)
    for (n, TᵀT) in zip(s.n, ie.TᵀT)
        Base.BLAS.axpy!(n, TᵀT, cov)
    end
    ## Tᵀf = Tᵀ(ie) * svec(s.f)
    Tᵀf = zeros(Float, nv)
    for (c, Tc) in enumerate(ie.T)
        Base.BLAS.gemm!('T', 'T', 1.0, Tc, sub(s.f, c, :), 1.0, Tᵀf)
    end
    if Linv
        L⁻¹ = inv(cov)
        w = L⁻¹ * Tᵀf
        Base.BLAS.gemm!('N', 'T', 1.0, w, w, 1.0, L⁻¹) ## L⁻¹ += w * w'
        return w, L⁻¹
    else
        w = cov \ Tᵀf
        return w
    end
end

## same for an array of stats
function posterior{Float<:AbstractFloat}(ie::IExtractor{Float}, s::Vector{CSstats{Float}})
    pmap(x->posterior(ie, x), s)
end

## update i-vector extractor according to the maximum likelihood re-estimation,
## ie: current i-vector extractor
## S: vector of Cstats (0th, 1st, 2nd order stats)
## post: vector of postriors / expectations, i.e., tuples E[y], E[y y']
function updateie!{Float<:AbstractFloat}(ie::IExtractor{Float}, S::Vector{CSstats{Float}}, post::Vector)
    @assert length(S) == length(post)
    ng, nfea, nv = size(ie)
    A = map(x -> zeros(Float, nv, nv), 1:ng)
    C = zeros(Float, ng * nfea, nv)
    for (s, p) in zip(S, post)         # loop over all utterances
        Ey, Eyyᵀ = p
        for c in 1:ng
            Base.LinAlg.BLAS.axpy!(s.n[c], Eyyᵀ, A[c]) # Eyyᵀ
        end
        # C += svec(s.f) * Ey
        Base.LinAlg.BLAS.gemm!('N', 'T', 1.0, svec(s.f), Ey, 1.0, C)
    end
    for c=1:ng
        range = ((c-1)*nfea+1) : c*nfea ## 1-based indexing, you are ugly!
        ie.T[c][:] = C[range,:] / A[c] ## C[range,:] * inv(A[c])
        Base.BLAS.gemm!('T', 'N', 1.0, ie.T[c], ie.T[c], 0.0, ie.TᵀT[c])  ## TᵀT[c] = T[c]' * T
    end
end

import GaussianMixtures.em!

function em!{Float<:AbstractFloat}(ie::IExtractor{Float}, S::Vector{CSstats{Float}}; nIter=1)
    for i=1:nIter
        print("Iteration ", i, "...")
        post = posterior(ie, S)
        updateie!(ie, S, post)
        println("done")
    end
    return ie
end

## Train an ivector extractor matrix
function IExtractor{Float<:AbstractFloat}(S::Vector{CSstats{Float}}, nvoices::Int; nIter=7)
    ng, nfea = size(first(S).f)
    T = map(i -> randn(nfea, nvoices), 1:ng)
    ie = IExtractor(T)
    for i=1:nIter
        print("Iteration ", i, "...")
        post = posterior(ie, S)
        updateie!(ie, S, post)
        println("done")
    end
    return ie
end

## extract an ivector using T-matrix and centered stats
function ivector{Float<:AbstractFloat}(ie::IExtractor{Float}, s::CSstats{Float})
    w = posterior(ie, s, Linv=false)
    return w
end

## extract multiple ivectors efficiently
function ivector{Float<:AbstractFloat}(ie::IExtractor{Float}, S::Vector{CSstats{Float}})
    nv = nvoices(ie)
    nfea, nutt = size(first(S).f, 2), length(S)
    covs = repmat(vec(eye(Float, nv)), 1, length(S)) ## nvoices^2 x nutt
    n = hcat([s.n for s in S]...) ## ng x nutt
    for (i, TᵀT) in enumerate(ie.TᵀT)
        Base.BLAS.gemm!('N', 'N', 1.0, vec(TᵀT), sub(n, i, :), 1.0, covs) # nvoices^2 x 1, 1 x nutt, nvoices^2 x nutt
    end
    ## Tᵀf = Tᵀ(ie) * svec(s.f)
    Tᵀfs = zeros(Float, nv, nutt) ## nvoices x nutt
    f = cat(3, [s.f for s in S]...)
    for (c, Tc) in enumerate(ie.T)
        fc = reshape(f[c, :, :], nfea, nutt)
        Base.BLAS.gemm!('T', 'N', 1.0, Tc, fc, 1.0, Tᵀfs) # nfea x nvoices, nfea x nutt, nvoices x nutt
    end
    map(i -> reshape(covs[:,i], nv, nv) \ reshape(Tᵀfs[:,i], nv), 1:nutt)
end
