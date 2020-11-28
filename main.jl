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
	name = "B/L3"
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
	plot(fluxes,frequencies,spectrum,targets)
	plot!(fluxes,ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels),linestyle=:dot,color=:white)
	plot!(titlefontsize=12,title=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\mathrm{GHz}\\quad E_C=$(round(parameters.Ec,digits=2))\\mathrm{GHz}\\quad E_J=$(round(parameters.Ej,digits=2))\\mathrm{GHz}\$")) |> display
	println(result)
end

# save final figure when happy
savefig(joinpath("figures",name*".pdf"))

begin # parameter uncertainty
	Elrange = range(0.01,1/2,length=50)
	Ejrange = range(0.01,3,length=50)

	contourf( Elrange, Ejrange, (x,y)->loss( fluxonium, merge(parameters,(El=x*parameters.Ec,Ej=y*parameters.Ec)), targets; nlevels=nlevels),
		size=(500,500), color=:tokyo, xlabel=L"E_L/E_C",ylabel=L"E_J/E_C", title=L"\mathrm{Loss\,\,Landscape}\quad\log L(\theta)")
	scatter!([parameters.El/parameters.Ec],[parameters.Ej/parameters.Ec],marker=:star,markersize=10,color=:white,label=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\mathrm{GHz}\\quad E_C=$(round(parameters.Ec,digits=2))\\mathrm{GHz}\\quad E_J=$(round(parameters.Ej,digits=2))\\mathrm{GHz}\$"))

end

# save final figure when happy
savefig(joinpath("figures",name*".uncertainty.pdf"))

##########################################################################
##########################################################################
################################################################## coupled

begin # load data with preprocessing parameters
	coupled_path = joinpath(name,"coupled")

	fluxes = 2π .* Load(joinpath("data",coupled_path,"fluxes.csv"))[:,1]
	frequencies = Load(joinpath("data",coupled_path,"frequencies.csv"))[:,1]
	targets = (fluxes=fluxes,frequencies=frequencies,weights=@. exp(-abs(sin(fluxes))))
	
	# show data
	plot( grid=false, ylim=(percentile(targets.frequencies,1),percentile(targets.frequencies,99)), size=(500,500), xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi", ylabel=L"\mathrm{Frequency\,\,\,GHz}")
	plot!( targets.fluxes, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels), color=cgrad(:tokyo)[end-4])
	scatter!( targets.fluxes, targets.frequencies, color=cgrad(:tokyo)[50], markerstrokewidth=0, markersize=4targets.weights, label="") |> display
end

begin # fit model parameters

	n = 5 # initialise resonator coupling hamiltonian
	resonator = I(N) ⊗ Resonator(n)
	system = Hermitian(zeros(n*N,n*N))

	############################ coupling terms
	a = annihilation(n) # resonator
	b = annihilation(N) # fluxonium

	inductive_term =  (b+b')⊗(a+a')/√2
	capacitive_term = (b-b')⊗(a-a')/√2

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
	plot( grid=false, ylim=(percentile(targets.frequencies,1),percentile(targets.frequencies,99)), size=(500,500), xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi", ylabel=L"\mathrm{Frequency\,\,\,GHz}")
	plot!( targets.fluxes, ϕ->Frequencies(fluxonium,ϕ,parameters;nlevels=nlevels), color=cgrad(:tokyo)[end-4])
	plot!( targets.fluxes, ϕ->Frequencies(system,ϕ,parameters;nlevels=nlevels_coupled,coupled=true); color=cgrad(:tokyo)[180] )
	scatter!( targets.fluxes, targets.frequencies, color=cgrad(:tokyo)[50], markerstrokewidth=0, markersize=4targets.weights, label="")
	plot!(titlefontsize=9,title=LaTeXString("\$E_L=$(round(parameters.El,digits=2))\\quad E_C=$(round(parameters.Ec,digits=2))\\quad E_J=$(round(parameters.Ej,digits=2))\\quad G_L=$(round(parameters.Gl,digits=2))\\quad G_C=$(round(parameters.Gc,digits=2))\\quad \\nu_R=$(round(parameters.νr,digits=2))\$")) |> display
	println(result)
end

# save final figure when happy
savefig(joinpath("figures",name*".coupled.pdf"))

begin # parameter uncertainty
	Gcrange = range(-0.8,0.8,length=50)
	Glrange = range(-0.2,0.2,length=50)

	contourf( Glrange, Gcrange, (x,y)->loss( system, merge(parameters,(Gl=x,Gc=y)), targets; nlevels=nlevels_coupled, coupled=true),
		size=(500,500), color=:tokyo, xlabel=L"\mathrm{Inductive\quad Coupling}\quad G_L",ylabel=L"\mathrm{Capacitive\quad Coupling}\quad G_C", title=L"\mathrm{Loss\,\,Landscape}\quad\log L(\theta)")
	scatter!([parameters.Gl],[parameters.Gc],marker=:star,markersize=10,color=:white,label=LaTeXString("\$G_L=$(round(parameters.Gl,digits=2))\\quad G_C=$(round(parameters.Gc,digits=2))\$")) |> display

end

# save final figure when happy
savefig(joinpath("figures",name*".coupled.uncertainty.pdf"))