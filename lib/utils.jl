using CSV,DataFrames,Plots,LaTeXStrings
using Images: imfilter,Kernel

using Serialization: serialize,deserialize
using Parameters: @unpack

function File(path; weight = ϕ->exp(-abs(sin(ϕ))), image_size=512, kwargs...)
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

    # resize spectrum to same aspect ratio
    spectrum = imresize(Gray.(spectrum),image_size,image_size)
    frequencies = range(minimum(frequencies),maximum(frequencies),length=image_size) |> collect
    fluxes = range(minimum(fluxes),maximum(fluxes),length=image_size) |> collect

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

"""
    edges = sujoy(img; four_connectivity=true)

Compute edges of an image using the Sujoy algorithm.

# Parameters

* `img` (Required): any gray image
* `four_connectivity=true`: if true, kernel is based on 4-neighborhood, else, kernel is based on
   8-neighborhood,

# Returns

* `edges` : gray image
"""
function sujoy(img; four_connectivity=true)
    img_channel = Gray.(img)

    min_val = minimum(img_channel)
    img_channel = img_channel .- min_val
    max_val = maximum(img_channel)

    if max_val == 0
        return img
    end

    img_channel = img_channel./max_val

    if four_connectivity
        krnl_h = centered(Gray{Float32}[0 -1 -1 -1 0; 0 -1 -1 -1 0; 0 0 0 0 0; 0 1 1 1 0; 0 1 1 1 0]./12)
        krnl_v = centered(Gray{Float32}[0 0 0 0 0; -1 -1 0 1 1;-1 -1 0 1 1;-1 -1 0 1 1;0 0 0 0 0 ]./12)
    else
        krnl_h = centered(Gray{Float32}[0 0 -1 0 0; 0 -1 -1 -1 0; 0 0 0 0 0; 0 1 1 1 0; 0 0 1 0 0]./8)
        krnl_v = centered(Gray{Float32}[0 0 0 0 0;  0 -1 0 1 0; -1 -1 0 1 1;0 -1 0 1 0; 0 0 0 0 0 ]./8)
    end

    grad_h = imfilter(img_channel, krnl_h')
    grad_v = imfilter(img_channel, krnl_v')

    grad = (grad_h.^2) .+ (grad_v.^2)

    return grad
end

import Plots: plot, plot!
function plot(fluxes::Vector,frequencies::Vector,spectrum::Array,targets::NamedTuple)
    plot( grid=false, size=(500,500), xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi", ylabel=L"\mathrm{Frequency\,\,\,GHz}",legend=:none)
    heatmap!(fluxes,frequencies,Float64.(spectrum),color=:inferno)

    scatter!(targets.fluxes, targets.frequencies, label="", alpha=0.5,
        color=:white, markerstrokewidth=0, markersize=7 .*targets.weights) |> display
end

function plot!(fluxes::Vector,frequencies::Vector,model::Function,parameters::NamedTuple; color=:gold)

    model_fluxes = minimum(fluxes):0.01:maximum(fluxes)
    model_frequencies = map(model,model_fluxes)

    for idx ∈ 1:length(first(model_frequencies))
        plot!( model_fluxes, map( ϕ->ϕ[idx], model_frequencies),
            label="", color=color, linewidth=3 )
    end

    plot!(titlefontsize=12,title=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\quad E_C=$(round(parameters.Ec,digits=2))\\quad E_J=$(round(parameters.Ej,digits=2))\\quad G_L=$(round(parameters.Gl,digits=2))\\quad G_C=$(round(parameters.Gc,digits=2))\$")) |> display
end