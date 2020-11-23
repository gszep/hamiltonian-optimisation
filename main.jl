begin # load required libraries
	using Optim,LinearAlgebra
	using Parameters: @unpack
	using LaTeXStrings,Plots

	include("lib/hamiltonian.jl")
	include("lib/optimisation.jl")

	include("lib/contours.jl")
	include("lib/utils.jl")
end

##########################################################################
##########################################################################
################################################################ uncoupled

begin # load data with preprocessing parameters
	name = "B/L4"
	data_path = joinpath("data",name)

	fluxes,frequencies,spectrum,targets = File(data_path;

		# preprocessing parameters can be updated here
		threshold=0.1, blur=0.5, downSample=100,
		#frequency_cutoff=Inf, flux_cutoff=Inf, maxTargets=1e3
	)
	plot(fluxes,frequencies,spectrum,targets)
end

begin # fit model parameters

	fluxonium = Hermitian(zeros(20,20))
	parameters = ( El=1.0,Ec=1.0,Ej=1.0, Gl=0.0,Gc=0.0 )
	nlevels = 1:3
	
	lower_bound, upper_bound = [0.0,0.0,0.0], [50.0,50.0,50.0]
	inital_guess = [1.0,1.0,1.0]
	
	# result = optimize(
	# 	x->loss(fluxonium, merge(parameters,(El=x[1],Ec=x[2],Ej=x[3])), targets; nlevels=nlevels),
	# 	lower_bound, upper_bound, inital_guess, Fminbox())
	
	# parameters = merge(parameters, (El=result.minimizer[1],Ec=result.minimizer[2],Ej=result.minimizer[3]) )
	parameters = merge(parameters, (El=1.7,Ec=0.4,Ej=10.3) )
	plot!( fluxes, frequencies, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels), parameters)
end

# save final figure when happy
savefig(joinpath("figures",replace(name,"/"=>"-")*".pdf"))

##########################################################################
##########################################################################
################################################################## coupled

begin # load data with preprocessing parameters
	name = "B/L4"
	data_path = joinpath("data",name)
	nlevels_uncoupled = 1

	fluxes,frequencies,spectrum,targets = File(data_path;

		#preprocessing parameters can be updated here
		# threshold=0.032, blur=1, downSample=10,
		# frequency_cutoff=6., flux_cutoff=-1, maxTargets=1e3
	)
	plot(  fluxes, frequencies, spectrum, targets )
	plot!( fluxes, frequencies, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels_uncoupled), parameters)
end

begin # fit model parameters

	resonator = Hermitian(zeros(5,5))
	parameters = merge(parameters,(νr=3.2,))

	system_size = size(resonator,1)*size(fluxonium,1)
	system = Hermitian(zeros(system_size,system_size))
	nlevels_coupled = 2

	lower_bound, upper_bound = [-0.1,-0.1], [0.1,0.1]
	inital_guess = [0.0,0.0]
	
	result = optimize(
		x->loss(system,fluxonium,resonator, merge(parameters,(Gl=x[1],Gc=x[2])), targets; nlevels=nlevels_coupled),
		lower_bound, upper_bound, inital_guess, Fminbox())

	parameters = merge(parameters, (Gl=0.1,Gc=0.1) ) # (Gl=result.minimizer[1],Gc=result.minimizer[2])

	plot(  fluxes, frequencies, spectrum, targets )
	plot!( fluxes, frequencies, ϕ->Frequencies(system,fluxonium,resonator,ϕ,parameters;nlevels=nlevels_coupled), parameters; color=:blue)
	plot!( fluxes, frequencies, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels_uncoupled), parameters)
end

# save final figure when happy
savefig(joinpath("figures",replace(name,"/"=>"-")*".pdf"))