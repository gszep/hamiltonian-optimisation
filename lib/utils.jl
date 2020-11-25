using CSV,DataFrames,Plots,LaTeXStrings
using Images: Gray,imfilter,Kernel,imresize,dilate,erode

using Serialization: serialize,deserialize
using Parameters: @unpack

function Load(source; kwargs...)
    if isfile(source)

        data = CSV.File(source; header=false, kwargs...) |> DataFrame
        return convert(Array,data)

    else 
        throw("no such file $source")
    end
end

import Base: merge
function merge(x::NamedTuple,y::AbstractVector)
    update = Dict( key=>y[i] for (i,key) ∈ enumerate(keys(x)[1:length(y)]) )
    return merge(x,update)
end

function File(path; weight = ϕ->exp(-abs(sin(ϕ))), image_size=512, kwargs...)
    parameters = Dict(

        :threshold => 0.2,
        :kernel => [-1 0 1; -2 0 2; -1 0 1],
        :dilations => 3, :erosions => 1,

        :dilationDirection => 1,
        :erosionDirection => 2,

        :maxTargets => 1e3, :downSample => 1,
        :frequencyLimits => (-Inf,Inf), :fluxLimits => (-Inf,Inf),
    )
    
    if isfile(joinpath(path,"parameters.jls"))
        merge!(parameters,open(deserialize,joinpath(path,"parameters.jls"),"r"))
    end
    merge!(parameters,kwargs)

    @unpack threshold,dilations,erosions,kernel,dilationDirection,erosionDirection,downSample,maxTargets,frequencyLimits,fluxLimits = parameters
    printstyled(color=:blue,bold=true,"""threshold\t$threshold\ndilations\t$dilations\nerosions\t$erosions\nkernel\t\t$kernel\n\ndilationDirection\t$dilationDirection\nerosionDirection\t$erosionDirection\n\nfrequencyLimits\t$frequencyLimits\nfluxLimits\t$fluxLimits\nmaxTargets\t$maxTargets\n""")

    # import csv files as arrays
    spectrum = Load(joinpath(path,"spectrum.csv"))
    frequencies = Load(joinpath(path,"frequencies.csv"))
    fluxes = Load(joinpath(path,"fluxes.csv"))

    # resize
    spectrum = imresize(spectrum,image_size,image_size)
    frequencies = range(minimum(frequencies),maximum(frequencies),length=image_size) |> collect
    fluxes = range(minimum(fluxes),maximum(fluxes),length=image_size) |> collect

    # apply cutoffs
    frequency_mask = @. (frequencyLimits[1]<frequencies) & (frequencies<frequencyLimits[2])
    flux_mask = @. (fluxLimits[1]<fluxes) & (fluxes<fluxLimits[2])

    spectrum = spectrum[frequency_mask,flux_mask]
    frequencies = frequencies[frequency_mask]
    fluxes = fluxes[flux_mask]

    ############################################### extract targets with edge detection
    edges = threshold .< imfilter(spectrum,kernel)
    mask = repeated_dilate(edges,dilations;region=dilationDirection)
    mask = repeated_erode(mask,erosions;region=erosionDirection)

    targets = (fluxes=Float64[],frequencies=Float64[], weights=Float64[])
    for index ∈ findall(mask)[1:downSample:end]
        frequencyIndex,fluxIndex = index[1],index[2]

        push!(targets.fluxes, fluxes[fluxIndex])
        push!(targets.frequencies, frequencies[frequencyIndex])
        push!(targets.weights, weight(fluxes[fluxIndex]) )
    end

    @assert(length(targets.fluxes)>0,"no data returned; decrease downSample or threshold")
    @assert(length(targets.fluxes)<maxTargets,"too much data; increase downSample or threshold")

    open(f -> serialize(f,parameters), joinpath(path,"parameters.jls"),"w")
    return fluxes,frequencies,spectrum,targets
end

repeated_dilate(img::AbstractArray, n::Integer; region=[1,2]) = reduce(∘, ntuple(_-> x->dilate(x,region), n))(img)
repeated_erode(img::AbstractArray, n::Integer; region=[1,2]) = reduce(∘, ntuple(_-> x->erode(x,region), n))(img)

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

    plot!(titlefontsize=9,title=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\quad E_C=$(round(parameters.Ec,digits=2))\\quad E_J=$(round(parameters.Ej,digits=2))\\quad G_L=$(round(parameters.Gl,digits=2))\\quad G_C=$(round(parameters.Gc,digits=2))\\quad \\nu_R=$(round(parameters.νr,digits=2))\$")) |> display
end