## ivector.jl  Various routines for ivector extraction
## (c) 2013--2014 David A. van Leeuwen

## Kenny-order supervectors correspond to GMM-order matrices by stacking row-wise, i.e., svec(X) = vec(X')
## for g in Gaussians for f in features X[g,f] end end
svec(x::Matrix) = vec(x')

## I + x, but doing I + x in-memory
function Iplusx!{T}(x::Matrix{T})
    for i in 1:size(x,1)
        x[i,i] += one(T)
    end
    return x
end

## this constructs the T matrix from the IExtractor vector-of-component-T's
Tᵀ(ie::IExtractor) = hcat([Tc' for Tc in ie.T]...)

## compute variance and mean of the posterior distribution of the hidden variables
## s: centered statistics (.N .F .S),
## v: svl x Nvoices matrix, initially random
## Σ: ng x nfea supervector diagonal covariance matrix, intially gmm.Σ
## result is a Nvoices vector and Nvoices x Nvoices matrix
function posterior{T<:AbstractFloat}(ie::IExtractor{T}, s::CSstats{T})
    nvoices = size(ie.T[1], 2)
    cov = eye(T, nvoices)
    for (n, TTᵀ) in zip(s.n, ie.TTᵀ)
        Base.LinAlg.BLAS.axpy!(n, TTᵀ, cov)
    end
    Linv = inv(cov)
    w = Linv * (Tᵀ(ie) * svec(s.f))                 # Nmul: svl * nv + nv^2
    return w, Linv + w * w'                             # nv and nv * nv
end

## same for an array of stats
function posterior{T}(ie::IExtractor, s::Vector{CSstats{T}})
    pmap(x->posterior(ie, x), s)
end

## update v and Σ according to the maximum likelihood re-estimation,
## S: vector of Cstats (0th, 1st, 2nd order stats)
## ex: vectopr of expectations, i.e., tuples E[y], E[y y']
## v: projection matrix
function updateie!{T<:AbstractFloat}(ie::IExtractor, S::Vector{CSstats{T}}, post::Vector)
    @assert length(S) == length(post)
    ng = length(ie.T)
    nfea, nv = size(ie.T[1])
    A = map(x -> zeros(nv, nv), 1:ng)
    C = zeros(ng * nfea, nv)
    for (s, p) in zip(S, post)         # loop over all utterances
        Ey, Eyyᵀ = p
        for c in 1:ng
            Base.LinAlg.BLAS.axpy!(s.n[c], Eyyᵀ, A[c]) # Eyyᵀ
        end
        # C += svec(s.F) * μ'          # Ey
        Base.LinAlg.BLAS.gemm!('N', 'T', 1.0, svec(s.f), Ey, 1.0, C)
    end
    for c=1:ng
        range = ((c-1)*nfea+1) : c*nfea
        ie.T[c][:] = C[range,:] / A[c] ## C[range,:] * inv(A[c])
        Base.BLAS.gemm!('T', 'N', 1.0, ie.T[c], ie.T[c], 0.0, ie.TTᵀ[c])  ## TTᵀ[c] = T[c]' * T
    end
end

import GaussianMixtures.em!

function em!{T1,T2}(ie::IExtractor{T1}, S::Vector{Cstats{T1,T2}}; nIter=1)
    v = ie.T
    ng = length(S[1].N)
    nfea = length(ie.Λ) ÷ ng
    Σ = reshape(ie.Λ, nfea, ng)'
    for i=1:nIter
        print("Iteration ", i, "...")
        ex = expectation(S, v)
        updatevΣ!(S, ex, v)
        println("done")
    end
    return ie
end

## Train an ivector extractor matrix
function IExtractor{T1<:AbstractFloat}(S::Vector{CSstats{T1}}, nvoices::Int; nIter=7)
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

# extract an ivector using T-matrix and uncentered stats
function ivector(ie::IExtractor, s::CSstats)
    w, = posterior(ie, s)
    return w
end
