using CSV,DataFrames,Plots,LaTeXStrings
using Images: imfilter,Kernel

using Serialization: serialize,deserialize
using Parameters: @unpack

function File(path; weight = ϕ->exp(-abs(sin(ϕ))), kwargs...)
    parameters = Dict(

        :threshold => 1, :blur => 1, :downSample => 1,
        :maxTargets => 1e3, :frequency_cutoff => Inf, :flux_cutoff => Inf
    )
    
    if isfile(joinpath(path,"parameters.jls")) merge!(parameters,open(deserialize,joinpath(path,"parameters.jls"),"r")) end
    merge!(parameters,kwargs)

    @unpack threshold,blur,downSample,maxTargets,frequency_cutoff,flux_cutoff = parameters
    printstyled(color=:blue,bold=true,"threshold\t$threshold\nblur\t\t$blur\ndownSample\t$downSample\n\nfrequency_cutoff\t$frequency_cutoff\nflux_cutoff\t\t$flux_cutoff\nmaxTargets\t\t$maxTargets\n")

    # import csv files as dataframes
    spectrum = isfile(joinpath(path,"spectrum.csv")) ? CSV.File(joinpath(path,"spectrum.csv"),header=false) |> DataFrame : throw("no such file $(joinpath(path,"spectrum.csv"))")
    frequencies = isfile(joinpath(path,"frequencies.csv")) ? CSV.File(joinpath(path,"frequencies.csv"),header=false) |> DataFrame : throw("no such file $(joinpath(path,"frequencies.csv"))")
    fluxes = isfile(joinpath(path,"fluxes.csv")) ? CSV.File(joinpath(path,"fluxes.csv"),header=false) |> DataFrame : throw("no such file $(joinpath(path,"fluxes.csv"))")

    # convert dataframes to mutlidimensional arrays
    spectrum = convert(Matrix,spectrum)
    frequencies = convert(Array,frequencies)[:,1]
    fluxes = convert(Array,fluxes)[:,1]

    ########################################### check dimensions match
    n,m = size(spectrum)
    N,M = length(frequencies),length(fluxes)

    @assert(m==M,"flux array length $M ≠ spectrum matrix width $M")
    @assert(n==N,"frequency array length $N ≠ spectrum matrix height $n")

    # apply cutoffs
    frequency_mask = frequencies .< frequency_cutoff
    flux_mask = fluxes .< flux_cutoff

    spectrum = spectrum[frequency_mask,flux_mask]
    frequencies = frequencies[frequency_mask]
    fluxes = fluxes[flux_mask]

    ############################################### extract targets with edge detection
    laplacian = imfilter(imfilter(spectrum, Kernel.gaussian(blur)), Kernel.Laplacian())
    mask = laplacian.<-abs(threshold)

    targets = (fluxes=Float64[],frequencies=Float64[], weights=Float64[])
    for index ∈  findall(mask)[1:downSample:end]

        push!(targets.fluxes, fluxes[index[2]])
        push!(targets.frequencies, frequencies[index[1]])
        push!(targets.weights, weight.(fluxes[index[2]]))
    end

    @assert(length(targets.fluxes)>0,"no data returned; decrease downSample or threshold")
    @assert(length(targets.fluxes)<maxTargets,"too much data; increase downSample or threshold")

    open(f -> serialize(f,parameters), joinpath(path,"parameters.jls"),"w")
    return fluxes,frequencies,spectrum,targets
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
