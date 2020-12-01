using Optim,LinearAlgebra

############################################################################
function Frequencies( hamiltonian::Hermitian, ϕ::T, parameters::NamedTuple; nlevels=1:3, coupled=false) where T<:Number

    if coupled
        Coupling!(hamiltonian,ϕ,parameters)
    else 
        Fluxonium!(hamiltonian,ϕ,parameters)
    end
    
    Δenergies = diff(eigvals!(hamiltonian,1:maximum(nlevels)+1))
    cumsum!(Δenergies,Δenergies)

    return Δenergies[nlevels]
end

function loss( hamiltonian::Hermitian, parameters::NamedTuple, data::NamedTuple; kwargs...)

    model(ϕ) = Frequencies(hamiltonian,ϕ,parameters; kwargs...)
    fmin,fmax = extrema(data.frequencies)
    frange = fmax - fmin

    weighted_errors = map(
        (ϕ,f,w) -> w * minimum( f̂ -> abs(f̂-f)/frange, model(ϕ) ),
        data.fluxes, data.frequencies, data.weights )

    return log(sum(weighted_errors)) - log(sum(data.weights))
end