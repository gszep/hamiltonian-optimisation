using LinearAlgebra
using Parameters: @unpack

using PhysicalConstants: CODATA2018
const ħ = Float64(CODATA2018.ħ).val


⊗(A,B) = kron(A,B) # define tensor product as kroneker product
function annihilation(n::Integer)
    return Bidiagonal(zeros(n), map(sqrt,1:n-1), :U)
end

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
    ω = √(8El*Ec)

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
            hamiltonian.data[i,j] = ω*(i-1/2) - cos(ϕ)*laguerre_term
        end
    end
end


function Resonator( N::Integer )
    return Diagonal(collect(1:N).-1/2)
end


function Coupling!( system::Hermitian, ϕ::T, parameters::NamedTuple) where T<:Number
    @unpack Gl,Gc,νr, El,Ec = parameters
    ϕ₀ = (8Ec/El)^(1/4)

    Fluxonium!(fluxonium,ϕ,parameters)
    system.data .= fluxonium ⊗ I(n)  +  νr*resonator  +  Gl*ϕ₀ * inductive_term - Gc/ϕ₀ * capacitive_term
    
    return nothing
end