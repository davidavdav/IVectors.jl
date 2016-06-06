## scoring.jl scoring methods for i-vectors
## (c) 2016 David A. van Leeuwen

import MLBase
import MultivariateStats

## cosine distance
function cosdist(i1::Vector, i2::Vector)
    n1 = norm(i1)
    n2 = norm(i2)
    if n1 == 0.0 || n2 == 0.0
        return 0
    else
        return i1 ⋅ i2 / (n1*n2)
    end
end

## ivectors as columns (dim-1) or rows (dim=2) in a matrix
function cosdist(i1::Matrix, i2::Matrix; dim=1)
    n1 = mapslices(norm, i1, dim)
    n2 = mapslices(norm, i2, dim)
    n1z = find(n1 .== 0.0)
    n2z = find(n2 .== 0.0)
    n1[n1z] = 1.0
    n2[n2z] = 1.0
    if dim==1
        return broadcast(/, i1, n1)' * broadcast(/, i2, n2)
    else
        return broadcast(/, i1, n1) * broadcast(/, i2, n2)'
    end
end

## or just a list of vectors
cosdist{T}(i1::Vector{Vector{T}}, i2::Vector{Vector{T}}) = cosdist(hcat(i1...), hcat(i2...))
