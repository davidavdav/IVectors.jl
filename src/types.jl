## IExtractor is a type that contains the information necessary for i-vector extraction:
## We now stor T as a vector of Nfea x Nvoices of length Ngaussians
## we reserve room for pre-multiplied TᵀT
type IExtractor{Float<:AbstractFloat}
    T::Vector{Matrix{Float}}
    TᵀT::Vector{Matrix{Float}}
    function IExtractor(T::Vector)
        length(T) > 0 || error("Empty vector of T-components")
        length(unique(map(size, T))) == 1 || error("Inconsistent matrix size in T-components")
        TᵀT = map(x -> x' * x, T)
        new(T, TᵀT)
    end
end
IExtractor{Float}(T::Vector{Matrix{Float}}) = IExtractor{Float}(T)

## a type representing the actual data, for serialization to file
type IExtractorSerializer{Float<:AbstractFloat}
    T::Vector{Matrix{Float}}
end
