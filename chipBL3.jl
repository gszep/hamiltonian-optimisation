### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 5c921290-24cb-11eb-3427-51bbe32cc4a2
begin
	using Optim,LaTeXStrings,Plots,LinearAlgebra
	include("lib/hamiltonian.jl");
end

# ╔═╡ a90b11f0-24ce-11eb-19c5-8565f74c3a2b
md"This is a snippet of text that helps me explain what I'm doing
$f(x)$
"

# ╔═╡ 0a7a3640-28c2-11eb-04a9-f17464e7b232
fluxes,frequencies,spectrum = File("./data/ChipB_L3/")

# ╔═╡ 3e48daee-24d8-11eb-29ef-c9da82b5f242
targets = preprocess(fluxes,frequencies,spectrum,
	threshold=0.002,blur=3,downSample=80);

# ╔═╡ b156f840-24cb-11eb-20e3-e36650eb73b0
begin
    plot( grid=false, size=(500,500),
		xlabel=L"\mathrm{External\,\,\,Phase}\,\,\,\phi",
		ylabel=L"\mathrm{Frequency\,\,\,GHz}",legend=:none)
    heatmap!(fluxes,frequencies,spectrum)

    scatter!(targets.fluxes, targets.frequencies, label="", alpha=0.5,
        color=:white, markerstrokewidth=0, markersize=7 .*targets.weights)
end

# ╔═╡ 2418da40-24d3-11eb-01f0-2513e0acaac3
begin
	H, nlevels = Hermitian(zeros(20,20)), 1
	result = optimize(
		x->loss(H,(El=x[1],Ec=x[2],Ej=x[3]), targets; nlevels=nlevels),
		zeros(3), 10ones(3), ones(3), Fminbox())
end

# ╔═╡ 5ae6126e-24da-11eb-0942-27d5c08836bb
begin
	model = (flux,parameters)->Frequencies(H,flux,parameters;nlevels=nlevels)
    parameters = round.(result.minimizer, digits=2)
    parameters = (El=parameters[1], Ec=parameters[2], Ej=parameters[3])

    model_fluxes = minimum(fluxes):0.01:maximum(fluxes)
    model_frequencies = map(x->model(x,parameters),model_fluxes)

    for idx ∈ 1:nlevels
        plot!( model_fluxes, map( x->x[idx], model_frequencies),
            label="", color=:gold, linewidth=3 )
    end

    plot!(title=LaTeXString(
			"\$E_L=$(parameters.El)
			\\quad E_C=$(parameters.Ec)
			\\quad E_J=$(parameters.Ej)\$"))
end

# ╔═╡ e4dbf430-24db-11eb-1432-317a9393536a
LaTeXString("\$E_L=$(parameters.El)
			\\quad E_C=$(parameters.Ec)
			\\quad E_J=$(parameters.Ej)\$")

# ╔═╡ 3069edc0-24d3-11eb-1aa6-23b2cf533c03
# begin 
# 	fluxes,frequencies,spectrum = File("./data/7e3/",frequency_cutoff=6.4)
# 	targets = preprocess(fluxes,frequencies,spectrum,
# 		threshold=0.008,blur=1,downSample=2)

# 	plot(fluxes,frequencies,spectrum,targets)
# end

# ╔═╡ eb770920-24cb-11eb-3fd8-b5be0b1501fa
# begin
# 	??, nlevels = Hermitian(zeros(20,20)), 2
# 	@time result = optimize(
# 		?->loss(??,(El=?[1],Ec=?[2],Ej=?[3]),targets;nlevels=nlevels),
# 		zeros(3), 10ones(3), ones(3), Fminbox())

# 	plot!(fluxes,frequencies,(?,?)->Frequencies(??,?,?;nlevels=nlevels),result)
# 	savefig("figures/7e3.pdf")
# end

# ╔═╡ Cell order:
# ╠═5c921290-24cb-11eb-3427-51bbe32cc4a2
# ╟─a90b11f0-24ce-11eb-19c5-8565f74c3a2b
# ╠═0a7a3640-28c2-11eb-04a9-f17464e7b232
# ╠═3e48daee-24d8-11eb-29ef-c9da82b5f242
# ╟─b156f840-24cb-11eb-20e3-e36650eb73b0
# ╠═2418da40-24d3-11eb-01f0-2513e0acaac3
# ╟─5ae6126e-24da-11eb-0942-27d5c08836bb
# ╟─e4dbf430-24db-11eb-1432-317a9393536a
# ╠═3069edc0-24d3-11eb-1aa6-23b2cf533c03
# ╠═eb770920-24cb-11eb-3fd8-b5be0b1501fa
