import JLD

JLD.readas(ies::IExtractorSerializer) = IExtractor(ies.T)
JLD.writeas(ie::IExtractor) = IExtractorSerializer(ie.T)

function FileIO.save(file::Union{AbstractString,IO}, ie::IExtractor)
    JLD.jldopen(file, "w") do fd
        JLD.addrequire(fd, IVectors)
        JLD.write(fd, "IExtractor", ie)
    end
end

FileIO.load(file::AbstractString, ::Type{IExtractor}) = load(File(format"IExtractor", file))
#FileIO.load(file, "IExtractor")

Base.eltype{T}(::FileIO.File{FileIO.DataFormat{T}}) = T

function detectiextractor(file::AbstractString)
    eltype(FileIO.query(file)) == :JLD && JLD.jldopen(file) do fd
        JLD.exists(fd, "IExtractor")
    end
end

FileIO.add_format(format"IExtractor", detectiextractor, ".iex", [:IVectors])

FileIO.load(file::File{format"IExtractor"}) = JLD.jldopen(filename(file)) do fd
    read(fd, "IExtractor")
end
