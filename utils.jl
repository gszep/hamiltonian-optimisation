using CSV,DataFrames
using Images: imfilter,Kernel

function laguerre(x::T, n::Integer, α::Integer) where T<:Number
    n == 0 ? (return one(T)) : nothing
    Lₖ₋₁, Lₖ = one(T), 1+α-x

    for k ∈ 1:n-1 # calculate using recurrance relation
        Lₖ₋₁, Lₖ = Lₖ, ( (2k+1+α-x)*Lₖ - (k+α)*Lₖ₋₁ ) / (k+1)
    end
    return Lₖ
end

function File(path; frequency_cutoff=8.0, flux_cutoff=Inf )

    spectrum = CSV.File(path*"spectrum.csv",header=false) |> DataFrame
    frequencies = CSV.File(path*"frequencies.csv",header=false) |> DataFrame
    fluxes = CSV.File(path*"fluxes.csv",header=false) |> DataFrame

    spectrum = convert(Matrix,spectrum)
    frequencies = convert(Array,frequencies)[:,1]
    fluxes = convert(Array,fluxes)[:,1] #2π.*

    frequency_mask = frequencies .< frequency_cutoff
    flux_mask = fluxes .< flux_cutoff

    spectrum = spectrum[frequency_mask,flux_mask]
    frequencies = frequencies[frequency_mask]
    fluxes = fluxes[flux_mask]

    return fluxes,frequencies,spectrum
end

function preprocess(fluxes,frequencies,spectrum; threshold=1, blur=1, downSample=1,
         weight=ϕ->exp.(-abs.(sin.(ϕ))) )

    laplacian = imfilter(imfilter(spectrum, Kernel.gaussian(blur)), Kernel.Laplacian())
    mask = laplacian.<-abs(threshold)

    data = (fluxes=Float64[],frequencies=Float64[], weights=Float64[])
    for index ∈  findall(mask)[1:downSample:end]

        push!(data.fluxes, fluxes[index[2]])
        push!(data.frequencies, frequencies[index[1]])
        push!(data.weights, weight(fluxes[index[2]]))
    end

    @assert(length(data.fluxes)≠0,"no data returned; decrease downSample or threshold")
    @assert(length(data.fluxes)<1e3,"too much data; increase downSample or threshold")
    return data
end
