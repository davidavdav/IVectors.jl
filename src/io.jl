import JLD

JLD.readas(ies::IExtractorSerializer) = IExtractor(ies.T)
JLD.writeas(ie::IExtractor) = IExtractorSerializer(ie.T)

function FileIO.save(file::Union{AbstractString,IO}, ie::IExtractor)
    JLD.jldopen(file, "w") do fd
        JLD.addrequire(fd, IVectors)
        JLD.write(fd, "IExtractor", ie)
    end
end

Base.eltype{format}(::FileIO.File{FileIO.DataFormat{format}}) = format

detectiextractor(file::AbstractString) = eltype(FileIO.query(file)) == :JLD && JLD.jldopen(file) do fd
    JLD.exists(fd, "IExtractor")
end

if ! (:IExtractor in keys(FileIO.sym2info))
    FileIO.add_format(format"IExtractor", detectiextractor, ".iex", [:IVectors])
end

FileIO.load(file::File{format"IExtractor"}) = JLD.jldopen(filename(file)) do fd
    read(fd, "IExtractor")
end

FileIO.load(file::AbstractString, ::Type{IExtractor}) = load(File(format"IExtractor", file))
#FileIO.load(file, "IExtractor")
