## ivector.jl  Various routines for ivector extraction
## (c) 2013--2014 David A. van Leeuwen

## Kenny-order supervectors correspond to GMM-order matrices by stacking row-wise, i.e., svec(X) = vec(X')
## for g in Gaussians for f in features X[g,f] end end
svec(x::Matrix) = vec(x')

## compute variance and mean of the posterior distribution of the hidden variables
## s: centered statistics (.N .F .S),
## v: svl x Nvoices matrix, initially random
## Σ: ng x nfea supervector diagonal covariance matrix, intially gmm.Σ
## result is a Nvoices vector and Nvoices x Nvoices matrix
function posterior{T<:AbstractFloat}(s::Cstats{T}, v::Matrix{T}, Σ::Matrix{T})
    svl, nv = size(v)
    @assert prod(size(s.F)) == prod(size(Σ)) == svl
    Nprec = svec(broadcast(/, s.N, Σ)) # use correct order in super vector
    cov = inv(eye(nv) + v' * broadcast(*, Nprec, v)) # inv(l), a nv * nv matrix
    Fprec =  svec(s.F ./ Σ)
    μ = cov * (v' * Fprec)
    μ, cov                              # nv and nv * nv
end

## compute expectations E[y] and E[y y']
function expectation(s::Cstats, v::Matrix, Σ::Matrix)
    Ey, cov = posterior(s, v, Σ)
    Eyyᵀ = Ey * Ey' + cov
    Ey, Eyyᵀ
end

## same for an array of stats
function expectation{T1,T2}(s::Vector{Cstats{T1,T2}}, v::Matrix, Σ::Matrix)
    map(x->expectation(x, v,  Σ), s)
end

## update v and Σ according to the maximum likelihood re-estimation,
## S: vector of Cstats (0th, 1st, 2nd order stats)
## ex: vectopr of expectations, i.e., tuples E[y], E[y y']
## v: projection matrix
function updatevΣ{T<:AbstractFloat,T2}(S::Vector{Cstats{T,T2}}, ex::Vector, v::Matrix)
    @assert length(S) == length(ex)
    ng, nfea = size(first(S).F)     # number of components or Gaussians
    svl = ng*nfea                   # supervector lenght, CF
    nv = length(first(ex)[1])       # number of voices
    N = zeros(ng)
    A = Any[zeros(nv,nv) for i=1:ng]
    C = zeros(svl, nv)
    for (s,e) in zip(S, ex)             # loop over all utterances
        n = s.N
        N += n
        for c=1:ng
            A[c] += n[c] * e[2]         # Eyyᵀ
        end
        C += svec(s.F) * e[1]'        # Ey
    end
    ## update v
    v = Array(T, svl, nv)
    for c=1:ng
        range = ((c-1)*nfea+1) : c*nfea
        v[range,:] = C[range,:] * inv(A[c])
    end
    ## update Σ
    Σ = -reshape(sum(C .* v, 2), nfea, ng)'    # diag(C * v')
    for s in S
        Σ += s.S
    end
    broadcast!(/, Σ, Σ, N)
    v, Σ
end

## Train an ivector extractor matrix
function IExtractor{T1<:AbstractFloat,T2}(ubm::GMM, S::Vector{Cstats{T1,T2}}, nvoices::Int; nIter=7, updateΣ=false)
    ng, nfea = size(first(S).F)
    v = randn(ng*nfea, nvoices) * sum(ubm.Σ) * 0.001
    Σ = ubm.Σ
    for i=1:nIter
        print("Iteration ", i, "...")
        ex = expectation(S, v, Σ)
        if updateΣ
            v, Σ = updatevΣ(S, ex, v)
        else
            v, = updatevΣ(S, ex, v)
        end
        println("done")
    end
    IExtractor(v, Σ)
end
## backwards compatibility
IExtractor{T1<:AbstractFloat,T2}(S::Vector{Cstats{T1,T2}}, ubm::GMM, nvoices::Int; nIter=7) = IExtractor(ubm, S, nvoices)

# extract an ivector using T-matrix and uncentered stats
function ivector(ie::IExtractor, s::Cstats)
    nv = size(ie.Tᵀ,1)
    ng = length(s.N)
    nfea = div(length(ie.prec), ng)
    TᵀΣF = ie.Tᵀ * (svec(s.F) .* ie.prec)
    Nprec = vec(broadcast(*, s.N', reshape(ie.prec, nfea, ng))) # Kenny-order
    w = inv(eye(nv) + ie.Tᵀ * broadcast(*, Nprec, ie.Tᵀ')) * TᵀΣF
end
