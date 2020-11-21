using CSV,DataFrames,Plots,LaTeXStrings
using Images: imfilter,Kernel

function File(path; frequency_cutoff=12.0, flux_cutoff=Inf )

    spectrum = CSV.File(path*"spectrum.csv",header=false) |> DataFrame
    frequencies = CSV.File(path*"frequencies.csv",header=false) |> DataFrame
    fluxes = CSV.File(path*"fluxes.csv",header=false) |> DataFrame

    spectrum = convert(Matrix,spectrum)
    frequencies = convert(Array,frequencies)[:,1]
    fluxes = convert(Array,fluxes)[:,1]

    n,m = size(spectrum)
    N,M = length(frequencies),length(fluxes)

    @assert(m==M,"flux array length $M ≠ spectrum matrix width $M")
    @assert(n==N,"frequency array length $N ≠ spectrum matrix height $n")

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

import Plots: plot, plot!
function plot(fluxes::Vector,frequencies::Vector,spectrum::Array,targets::NamedTuple)
    plot( grid=false, size=(500,500), xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi", ylabel=L"\mathrm{Frequency\,\,\,GHz}",legend=:none)
    heatmap!(fluxes,frequencies,spectrum)

    scatter!(targets.fluxes, targets.frequencies, label="", alpha=0.5,
        color=:white, markerstrokewidth=0, markersize=7 .*targets.weights) |> display
end

function plot!(fluxes::Vector,frequencies::Vector,model::Function,result)

    parameters = round.(result.minimizer, digits=2)
    parameters = (El=parameters[1], Ec=parameters[2], Ej=parameters[3])

    model_fluxes = minimum(fluxes):0.01:maximum(fluxes)
    model_frequencies = map(ϕ->model(ϕ,parameters),model_fluxes)

    for idx ∈ 1:nlevels
        plot!( model_fluxes, map( ϕ->ϕ[idx], model_frequencies),
            label="", color=:gold, linewidth=3 )
    end

    plot!(title=LaTeXString("\$E_L=$(parameters.El)\\quad E_C=$(parameters.Ec)\\quad E_J=$(parameters.Ej)\$")) |> display
end
