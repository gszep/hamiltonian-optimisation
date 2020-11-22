begin # load required libraries
	using Optim,LinearAlgebra
	using Parameters: @unpack
	using LaTeXStrings,Plots

	include("lib/hamiltonian.jl")
	include("lib/optimisation.jl")
	include("lib/utils.jl")
end

begin # load data with preprocessing parameters
	name = "C/L4"
	data_path = joinpath("data",name)

	fluxes,frequencies,spectrum,targets = File(data_path;

		# threshold=0.01, blur=1, downSample=10,
		# frequency_cutoff=9.5, flux_cutoff=Inf, maxTargets=1e3
	)
	plot(fluxes,frequencies,spectrum,targets)
end

begin # fit model parameters

    H = Hermitian(zeros(20,20))
	nlevels = 1
	
	lower_bound, upper_bound = zeros(3), 10ones(3)
	inital_guess = ones(3)
	
	result = optimize(
		x->loss(H, (El=x[1],Ec=x[2],Ej=x[3]), targets; nlevels=nlevels),
		lower_bound, upper_bound, inital_guess, Fminbox())

	plot!( fluxes, frequencies, (ϕ,parameters)->Frequencies(H,ϕ,parameters;nlevels=nlevels), result)
end

# save final figure when happy
savefig(joinpath("figures",replace(name,"/"=>"-")*".pdf"))