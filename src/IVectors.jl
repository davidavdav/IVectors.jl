module IVectors

using GaussianMixtures
using FileIO

export svec, IExtractor, ivector
include("types.jl")
include("ivector.jl")
include("io.jl")

end
