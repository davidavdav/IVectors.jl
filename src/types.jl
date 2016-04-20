## Iextractor is a type that contains the information necessary for i-vector extraction:
## The T-matrix and an updated precision matrix prec
## It is difficult to decide how to store T and Σ, as T' and vec(prec)?
type IExtractor{T<:AbstractFloat}
    Tᵀ::Matrix{T}
    prec::Vector{T}
    function IExtractor(Tee::Matrix, prec::Vector)
        @assert size(Tee,1) == length(prec)
        new(Tee', prec)
    end
end
IExtractor{T<:AbstractFloat}(Tee::Matrix{T}, prec::Vector{T}) = IExtractor{T}(Tee, prec)
## or initialize with a traditional covariance matrix
IExtractor{T<:AbstractFloat}(Tee::Matrix{T}, Σ::Matrix{T}) = IExtractor{T}(Tee, vec(1./Σ'))
