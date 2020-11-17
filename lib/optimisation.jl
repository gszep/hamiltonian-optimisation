using Optim,LinearAlgebra

function Frequencies( hamiltonian::Hermitian; nlevels=3, νr=nothing, Δν=1.0)
    if isnothing(νr)
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