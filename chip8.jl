begin
	using Optim,LinearAlgebra
	using Parameters: @unpack
	using LaTeXStrings,Plots

	include("lib/hamiltonian.jl")
	include("lib/optimisation.jl")
	include("lib/utils.jl")
end

fluxes,frequencies,spectrum = File("./data/Chip8/"; frequency_cutoff=15.0)
fluxes .-= 0.05

targets = preprocess(fluxes,frequencies,spectrum,
	threshold=0.0014,blur=3,downSample=50)

plot(fluxes,frequencies,spectrum,targets)

begin
	H, nlevels = Hermitian(zeros(20,20)), 1
	result = optimize(
		x->loss(H,(El=x[1],Ec=x[2],Ej=x[3]), targets; nlevels=nlevels),
		zeros(3), 10ones(3), ones(3), Fminbox())
end

plot!(fluxes,frequencies,
	(flux,parameters)->Frequencies(H,flux,parameters;nlevels=nlevels),
result)

savefig("figures/Chip8.pdf")