using Optim,LinearAlgebra

############################################################################ uncoupled
function Frequencies( hamiltonian::Hermitian, ϕ::T, parameters::NamedTuple; nlevels=3) where T<:Number
    Fluxonium!(hamiltonian,ϕ,parameters)
    return cumsum(diff(eigvals(hamiltonian,1:nlevels+1)))
end

function loss( hamiltonian::Hermitian, parameters::NamedTuple, data::NamedTuple; nlevels=3)
    frequencies = hcat( Frequencies.( Ref(hamiltonian), data.fluxes, Ref(parameters); nlevels=nlevels)... )
    least_squares = ( data.frequencies .- frequencies' ).^2
    log(first(data.weights'minimum(least_squares,dims=2))) - log(sum(data.weights))
end

############################################################################ coupled
function Frequencies( system::Hermitian, fluxonium::Hermitian, resonator::Hermitian, ϕ::T, parameters::NamedTuple; nlevels=2) where T<:Number
    Coupling!(system,fluxonium,resonator,ϕ,parameters)
    return cumsum(diff(eigvals(system,1:nlevels+1)))
end

function loss( system::Hermitian, fluxonium::Hermitian, resonator::Hermitian, parameters::NamedTuple, data::NamedTuple; nlevels=2)
    frequencies = hcat( Frequencies.( Ref(system), Ref(fluxonium), Ref(resonator), data.fluxes, Ref(parameters); nlevels=nlevels)... )
    least_squares = ( data.frequencies .- frequencies' ).^2
    log(first(data.weights'minimum(least_squares,dims=2))) - log(sum(data.weights))
end