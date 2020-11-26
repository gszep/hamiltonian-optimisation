begin # load required libraries
	using Optim,LinearAlgebra,StatsBase
	using Parameters: @unpack
	using LaTeXStrings,Plots

	include("lib/hamiltonian.jl")
	include("lib/optimisation.jl")
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
		# threshold = 0.047, downSample=10, dilations=3, erosions=1,
		# frequencyLimits=(-Inf,9.5)
	)
	plot(fluxes,frequencies,spectrum,targets)
end

begin # fit model parameters

	N = 20 # initialise fluxonium hamiltonian
	fluxonium = Hermitian(zeros(N,N))

	parameters = ( El=1.0,Ec=1.0,Ej=1.0, Gl=0.0,Gc=0.0,νr=NaN )
	nlevels = 1:1
	
	##################################### optimisation
	result = optimize(

		x->loss(fluxonium, merge(parameters,(El=x[1],Ec=x[2],Ej=x[3])),
		targets; nlevels=nlevels),

		[ parameters.El, parameters.Ec, parameters.Ej ],
		NelderMead(), Optim.Options(iterations=10^4))
	
	# update parameters
	fluxonium_parameters = merge((El=NaN,Ec=NaN,Ej=NaN),result.minimizer)
	parameters = merge(parameters,fluxonium_parameters)

	# show results
	plot!( fluxes, frequencies, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels), parameters)
	println(result)
end

# save final figure when happy
savefig(joinpath("figures",name*".pdf"))

##########################################################################
##########################################################################
################################################################## coupled

begin # load data with preprocessing parameters
	coupled_path = joinpath(name,"coupled")

	fluxes = 2π .* Load(joinpath("data",coupled_path,"fluxes.csv"))[:,1]
	frequencies = Load(joinpath("data",coupled_path,"frequencies.csv"))[:,1]
	targets = (fluxes=fluxes,frequencies=frequencies,weights=@. exp(-abs(sin(fluxes))))
	
	# show data
	plot( grid=false, size=(500,500), xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi", ylabel=L"\mathrm{Frequency\,\,\,GHz}")
	scatter!( targets.fluxes, targets.frequencies, color=:darkblue, ylim=(percentile(targets.frequencies,1),percentile(targets.frequencies,99)), markerstrokewidth=0, markersize=3targets.weights, label="Coupling Data")
	plot!( targets.fluxes, targets.frequencies, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels), parameters)
end

begin # fit model parameters

	n = 2 # initialise resonator coupling hamiltonian
	resonator = I(N) ⊗ Resonator(n)
	system = Hermitian(zeros(n*N,n*N))

	############################ coupling terms
	a = annihilation(n) # resonator
	b = annihilation(N) # fluxonium

	inductive_term =  (b'+b)⊗(a+a')/√2
	capacitive_term = (b'-b)⊗(a-a')/√2

	##################################### optimisation
	parameters = merge(parameters,( Gl=-0.02,Gc=0.331,νr=5.9515 ))
	nlevels_coupled = 1:2

	result = optimize(

		x->loss(system, merge(parameters,(Gl=x[1],Gc=x[2],νr=x[3])),
		targets; nlevels=nlevels_coupled, coupled=true),

		[ parameters.Gl, parameters.Gc, parameters.νr ],
		NelderMead(), Optim.Options(iterations=10^4)
	)

	# update parameters
	coupling_parameters = merge((Gl=NaN,Gc=NaN,νr=NaN),result.minimizer)
	parameters = merge(parameters,coupling_parameters)

	# show results
	plot!( fluxes, frequencies, ϕ->Frequencies(system,ϕ,parameters;nlevels=nlevels_coupled,coupled=true), parameters; color=:blue)
	println(result)
end

# save final figure when happy
savefig(joinpath("figures",name*".coupled.pdf"))

begin # explore coupling parameter uncertainty
	Gcrange = range(-0.8,0.8,length=50)
	Glrange = range(-0.1,0.1,length=50)

	contourf( Glrange, Gcrange, (x,y)->loss( system, merge(parameters,(Gl=x,Gc=y)), targets; nlevels=nlevels_coupled, coupled=true),
		size=(500,500), xlabel=L"\mathrm{Inductive\quad Coupling}\quad G_L",ylabel=L"\mathrm{Capacitive\quad Coupling}\quad G_C")
	plot!(titlefontsize=10,title=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\quad E_C=$(round(parameters.Ec,digits=2))\\quad E_J=$(round(parameters.Ej,digits=2))\\quad \\nu_R=$(round(parameters.νr,digits=2))\$"))
	scatter!([parameters.Gl],[parameters.Gc],label=LaTeXString("\$G_L=$(round(parameters.Gl,digits=2))\\quad G_C=$(round(parameters.Gc,digits=2))\$"))

end

# save final figure when happy
savefig(joinpath("figures",name*".coupled.uncertainty.pdf"))