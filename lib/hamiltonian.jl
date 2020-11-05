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

    # populate upper triangle j ≥ i
    for i ∈ 0:N-1, j ∈ i:N-1
        Laguerre = exp(-ϕ₀^2/4) * laguerre(ϕ₀^2/2,min(i,j),abs(i-j))

        if i ≠ j
            normedLaguerre = ϕ₀^abs(i-j) * Laguerre / ( 2^((max(i,j)-min(i,j))/2) * prod( sqrt, min(i+1,j+1):max(i,j) ))

            if (i+j) % 2 == 0
                hamiltonian.data[i+1,j+1] = - Ej*cos(ϕ) * (-1)^( abs(i-j)/2   ) * normedLaguerre
            else
                hamiltonian.data[i+1,j+1] = - Ej*sin(ϕ) * (-1)^((abs(i-j)-1)/2) * normedLaguerre
            end
        else
            hamiltonian.data[i+1,j+1] = √(8El*Ec)*(i+1/2) - Ej*cos(ϕ) * Laguerre
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
