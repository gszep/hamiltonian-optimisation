using Optim
include("lib/hamiltonian.jl")

#########################################################################
#########################################################################
#########################################################################
fluxes,frequencies,spectrum = File("./data/6cb/")
targets = preprocess(fluxes,frequencies,spectrum,
    threshold=0.025,blur=3,downSample=10)

plot(fluxes,frequencies,spectrum,targets)

#########################################################################
𝓗, nlevels = Hermitian(zeros(20,20)), 1
@time result = optimize(
    θ->loss(𝓗,(El=θ[1],Ec=θ[2],Ej=θ[3]),targets;nlevels=nlevels),
    zeros(3), 10ones(3), ones(3), Fminbox())

plot!(fluxes,frequencies,(ϕ,θ)->Frequencies(𝓗,ϕ,θ;nlevels=nlevels),result)

#########################################################################
#########################################################################
#########################################################################
fluxes,frequencies,spectrum = File("./data/7e3/",frequency_cutoff=6.4)
targets = preprocess(fluxes,frequencies,spectrum,
    threshold=0.008,blur=1,downSample=2)

plot(fluxes,frequencies,spectrum,targets)

#########################################################################
𝓗, nlevels = Hermitian(zeros(20,20)), 2
@time result = optimize(
    θ->loss(𝓗,(El=θ[1],Ec=θ[2],Ej=θ[3]),targets;nlevels=nlevels),
    zeros(3), 10ones(3), ones(3), Fminbox())

plot!(fluxes,frequencies,(ϕ,θ)->Frequencies(𝓗,ϕ,θ;nlevels=nlevels),result)
