using LinearAlgebra
using Parameters: @unpack
include("utils.jl")

function laguerre(x::T, n::Integer, α::Integer) where T<:Number
    n == 0 ? (return one(T)) : nothing
    Lₖ₋₁, Lₖ = one(T), 1+α-x

    for k ∈ 1:n-1 # calculate using recurrance relation
        Lₖ₋₁, Lₖ = Lₖ, ( (2k+1+α-x)*Lₖ - (k+α)*Lₖ₋₁ ) / (k+1)
    end
    return Lₖ
end

function Fluxonium!( hamiltonian::Hermitian, ϕ::T, parameters::NamedTuple) where T<:Number
    @unpack El, Ec, Ej = parameters
    N,N = size(hamiltonian)

    ϕ₀ = (8Ec/El)^(1/4)
    ħω = √(8El*Ec)

    laguerre_multiplier = Ej*exp(-ϕ₀^2/4)
    for i ∈ 1:N, j ∈ i:N # populate upper triangle j ≥ i
        laguerre_term = laguerre_multiplier * laguerre(ϕ₀^2/2,min(i,j)-1,abs(i-j))

        if i ≠ j # off diagonal terms
            laguerre_term *= ϕ₀^abs(i-j) / ( 2^((max(i,j)-min(i,j))/2) * prod( sqrt, min(i,j):max(i,j)-1 ))

            if (i+j) % 2 == 0 # even terms
                hamiltonian.data[i,j] = -(-1)^( abs(i-j)/2   ) * cos(ϕ)*laguerre_term
            else # odd terms
                hamiltonian.data[i,j] = -(-1)^((abs(i-j)-1)/2) * sin(ϕ)*laguerre_term
            end

        else # diagonal terms
            hamiltonian.data[i,j] = ħω*(i-1/2) - cos(ϕ)*laguerre_term
        end
    end
end

function Frequencies( hamiltonian::Hermitian, ϕ::T, parameters::NamedTuple; nlevels=3 ) where T<:Number
    Fluxonium!( hamiltonian, ϕ, parameters )
    return cumsum(diff(eigvals!(hamiltonian,1:nlevels+1)))
end

function loss( hamiltonian::Hermitian, parameters::NamedTuple, data::NamedTuple; nlevels=3) where T<:Number
    frequencies = hcat( Frequencies.( Ref(hamiltonian), data.fluxes, Ref(parameters); nlevels=nlevels)... )
    least_squares = ( data.frequencies .- frequencies' ).^2
    log(first(data.weights'minimum(least_squares,dims=2))) - log(sum(data.weights))
end
