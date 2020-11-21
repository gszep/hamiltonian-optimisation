using Optim,LinearAlgebra

function Frequencies( hamiltonian::Hermitian, ϕ::T, parameters::NamedTuple;
        nlevels=3, νr=nothing, Δν=1.0) where T<:Number

    if isnothing(νr)
        Fluxonium!(hamiltonian,ϕ,parameters)
        return cumsum(diff(eigvals(hamiltonian,1:nlevels+1)))
    else
        return eigvals(hamiltonian)
    end
end

function loss( hamiltonian::Hermitian, parameters::NamedTuple, data::NamedTuple; nlevels=3)
    frequencies = hcat( Frequencies.( Ref(hamiltonian), data.fluxes, Ref(parameters); nlevels=nlevels)... )
    least_squares = ( data.frequencies .- frequencies' ).^2
    log(first(data.weights'minimum(least_squares,dims=2))) - log(sum(data.weights))
end