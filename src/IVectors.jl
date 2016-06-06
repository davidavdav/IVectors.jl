module IVectors

using GaussianMixtures
using FileIO

export IExtractor, ivector

include("types.jl")
include("ivector.jl")
include("scoring.jl")
include("io.jl")

end
