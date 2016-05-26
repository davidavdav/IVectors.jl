## Iextractor is a type that contains the information necessary for i-vector extraction:
## We now stor T as a vector of Nfea x Nvoices of length Ngaussians
## we reserve room for pre-multiplied TTᵀ
type IExtractor{F<:AbstractFloat}
    T::Vector{Matrix{F}}
    TTᵀ::Vector{Matrix{F}}
    function IExtractor(T::Vector)
        TTᵀ = map(x -> x' * x, T)
        new(T, TTᵀ)
    end
end
IExtractor{F}(T::Vector{Matrix{F}}) = IExtractor{F}(T)
