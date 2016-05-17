## Iextractor is a type that contains the information necessary for i-vector extraction:
## The T-matrix and an updated precision matrix prec
## It is difficult to decide how to store T and Σ, as T' and vec(Λ)?
type IExtractor{T<:AbstractFloat}
    Tᵀ::Matrix{T}
    Λ::Vector{T} ## Kenny-order precision, "gaussian index"-major
    function IExtractor(Tee::Matrix, Λ::Vector)
        size(Tee,1) == length(Λ) || throw(DimensionMismatch)
        new(Tee', Λ)
    end
end
IExtractor{T<:AbstractFloat}(Tee::Matrix{T}, prec::Vector{T}) = IExtractor{T}(Tee, prec)
## or initialize with a traditional covariance matrix
IExtractor{T<:AbstractFloat}(Tee::Matrix{T}, Σ::Matrix{T}) = IExtractor{T}(Tee, vec(1./Σ'))
