#==========================================================================================
Title: Plots
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-10-30
Description: This file contains the code to generate the plots of the paper
==========================================================================================#
@everywhere using CairoMakie
@everywhere using MathTeXEngine
@everywhere using LaTeXStrings
@everywhere using ColorSchemes

colors = ["#23363B", "#EB811A"]
#?=========================================================================================
#? I create a type to store all the figures and functions to plot them in a single place
#?=========================================================================================
#==========================================================================================
# Figres: Type to store all the figures 
==========================================================================================#
mutable struct Figures
    dist_types          ::Figure
    dist_unemp          ::Figure
    dist_match          ::Figure
    S_surv              ::Figure
    poaching_rank       ::Figure
    vacancies           ::Figure
    eq_vacancies        ::Figure
    agg_dist            ::Figure
    search_unemp        ::Figure
    search_emp          ::Figure


    # Constructor
    function Figures(resolution = (600, 400), fonts = (; regular= texfont(), bold = regular= texfont()))
        ℓ = Figure(resolution = resolution, fonts = fonts)
        u = Figure(resolution = resolution, fonts = fonts)
        h = Figure(resolution = resolution, fonts = fonts)
        S_surv = Figure(resolution = resolution, fonts = fonts)
        poaching_rank = Figure(resolution = resolution, fonts = fonts)
        vacancies = Figure(resolution = resolution, fonts = fonts)
        eq_vacancies = Figure(resolution = resolution, fonts = fonts)
        agg_dist = Figure(resolution = resolution, fonts = fonts)
        search_unemp = Figure(resolution = resolution, fonts = fonts)
        search_emp = Figure(resolution = resolution, fonts = fonts)
        new(ℓ, u, h, S_surv, poaching_rank, vacancies, eq_vacancies, agg_dist, search_unemp, search_emp)
    end # constructor
end # Figures
#==========================================================================================
# save_figure: Function to save a particular figure 
==========================================================================================#
function save_figure(figures::Figures, figure::Symbol; name::String = "", format = "pdf" )
    if name == ""
        name = string(figure)
    end
    save("./$(figures_dir)/$(name).$(format)", getfield(figures, figure))
    # println("Saving figure $(name)")
end # save_figure
#==========================================================================================
# save_all_figures: Function to save all the figures in the Figures type
==========================================================================================#
function save_all_figures(figures::Figures; prefix::String= "", sufix::String="", format = "pdf")
    # Get field names
    fields = fieldnames(Figures)
    for field in fields
        save_figure(figures, field, name = prefix * string(field) * sufix, format = format)
    end
end # save_all_figures
#?=========================================================================================
#? Generate figures for the paper
#?=========================================================================================
#==========================================================================================
# generate_colors: Function to generate a vector of colors from a colormap
==========================================================================================#
function generate_colors(cmap::Symbol, N::Int64)
    cmap = :RdYlBu
    # Extract N equally spaced colors from the color map
    if N > 1
        colors = get.( Ref(ColorSchemes.colorschemes[cmap]), range(0.1, 0.9, length = N))
    else
        colors = [get(ColorSchemes.colorschemes[cmap], 0.1)]
    end
    return colors
end
#==========================================================================================
# plot_distribution_2D: Function to plot a 2D distribution (types or unemployment)
==========================================================================================#
function plot_distribution_2D(prim::Primitives, dist_vector::Array{Float64, 2}, sizes;
                            title::String = "", xlabel = "Worker Type" , labels = [],
                            cmap = :plasma , backgroundcolor = "#FAFAFA")
    @unpack n_x, x_grid, n_j = prim
    
    fig = Figure(resolution = (600, 400), fonts = (; regular= texfont(), bold = regular= texfont()), 
                backgroundcolor = backgroundcolor)

    ax = Axis(fig[1,1], xlabel = xlabel, ylabel = "", title = title,
        ylabelsize = 18, xticks = 0:0.25:1,  titlesize = 22,
        xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
        xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    hidespines!(ax, :t, :r)
    # colors = generate_colors(cmap, n_j)
    if length(labels) == 0
        labels = ["Location $(j) Size: $(round( sizes[j], digits =  2) )" for j ∈ 1:n_j]
    end
    n_elements = size(dist_vector, 1)
    for j ∈ 1:n_elements
        lines!(x_grid, dist_vector[j, :]; label = labels[j] , linewidth = 4, color = colors[j])
        # lines!(x_grid, dist_vector[j, :]; label = labels[j] , linewidth = 4)
    end

    axislegend(""; position = :rt, bgcolor = (:grey90, 0.25));

    return fig
end
#==========================================================================================
# plot_distribution_3D: Function to plot a 3D distribution (matches) 
==========================================================================================#
function plot_distribution_3D(prim::Primitives, dist_vector::Array{Float64, 3}, sizes;
                            title::String = "", xlabel = "Worker Type", ylabel = "Firm Type",
                            cmap = :plasma , backgroundcolor = "#FAFAFA")

    @unpack n_x, x_grid, n_y, y_grid, n_j = prim

    zmin, zmax = minimum(dist_vector), maximum(dist_vector)

    fig = Figure(resolution = (n_j .* 400, 450), fonts = (; regular= texfont(), bold = regular= texfont()), 
                backgroundcolor = backgroundcolor)
    axs = [Axis(fig[1, j], xlabel = xlabel, ylabel = (j == 1) ? ylabel : "",
    ylabelsize = 18,  titlesize = 22,
    xticks = 0:0.25:1, yticks = 0.25:0.25:1, title = "Location $(j)",
    xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0) for j ∈ 1:size(dist_vector, 1)]

    for j ∈ 1:size(dist_vector, 1)
        z = dist_vector[j,:,:]
        CairoMakie.heatmap!(axs[j], x_grid, y_grid, z, 
        colormap = (cmap,  0.85), clim = (zmin, zmax), 
        colorrange = (zmin, zmax), colorbar = j == size(dist_vector, 1))
        CairoMakie.contour!(axs[j],  x_grid, y_grid, z .> 0,; color=:white, levels=1, linewidth=3)
    end
    Label(fig[0, :], text = title, fontsize = 32, halign = :center, valign = :center)
    return fig
    # Colorbar(fig[1, n_j], width=20, ticksize=20, tickalign=1)
    # [limits!(axs[i], 1, 20, 1, 20) for i in 1:2]
    # [hideydecorations!(axs[i], grid=false, ticks=false) for i in 2:3]
end 
#==========================================================================================
# plot_equilibrium_distribution: Function to plot and store the equilibrium distributions 
==========================================================================================#
function plot_equilibrium_distribution( prim::Primitives, dist::DistributionsModel, figures::Figures;
                                xlabel = "Worker Type", ylabel = "Firm Type", cmap = :plasma , backgroundcolor = "#FAFAFA")
    
    sizes = round.(sum(dist.ℓ, dims=2), digits=4)[:]

    fig_ℓ = plot_distribution_2D(prim, dist.ℓ, sizes, xlabel = xlabel, title = "", cmap = cmap, backgroundcolor = backgroundcolor)
    fig_u = plot_distribution_2D(prim, dist.u_plus, sizes, xlabel = xlabel, title = "", cmap= cmap, backgroundcolor = backgroundcolor)
    fig_h = plot_distribution_3D(prim, dist.h, sizes, xlabel = xlabel, ylabel = ylabel, title = "", cmap= cmap, backgroundcolor = backgroundcolor)

    # Keep track of figures
    figures.dist_types = fig_ℓ
    figures.dist_unemp = fig_u
    figures.dist_match = fig_h

end
#==========================================================================================
# plot_surviving_matches: Function to plot and store the mathces that survive endogenous
                    detruction
==========================================================================================#
function plot_surviving_matches(prim::Primitives, res::Results, figures::Figures; 
    title::String = "", xlabel = "Worker Type", ylabel = "Firm Type", cmap = :plasma , backgroundcolor = "#FAFAFA")

    @unpack n_j, x_grid, y_grid = prim
    @unpack S_move, y_star = res

    fig = Figure(resolution = (n_j .* 400, 450), fonts = (; regular= texfont(), bold = regular= texfont()), 
                backgroundcolor = backgroundcolor)


    for j ∈ 1:n_j
        j_prime = j
        ax =Axis(fig[1, j],  xlabel = xlabel, ylabel = (j_prime == 1) ? ylabel : "",
            ylabelsize = 18,  titlesize = 22,
            xticks = 0:0.25:1, yticks = 0.25:0.25:1, title = "Location $(j)",
            xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
            xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0) 

        CairoMakie.heatmap!(ax, x_grid, y_grid, S_move[j,j_prime,:,:] .> 0,  colormap = cmap)
        lines!(ax, x_grid, y_star[j_prime, :], label = "", linewidth = 3, color = :black, linestyle = :dash)
    end
    Label(fig[0, :], text = title, fontsize = 32, halign = :center, valign = :center)

    figures.S_surv = fig
end
#==========================================================================================
# plot_lines_mode: Function to plot and store the lines of the model (poaching index, 
                    vacancies, etc.)
==========================================================================================#
function plot_lines_model(prim::Primitives, lines;
    title::String = "", xlabel = "" , cmap = :plasma , backgroundcolor = "#FAFAFA")
    
    @unpack y_grid, x_grid, n_j = prim

    fig = Figure(resolution = (600, 400), fonts = (; regular= texfont(), bold = regular= texfont()))

    ax = Axis(fig[1,1], xlabel = xlabel, ylabel = "", title = title,
    ylabelsize = 18, xticks = 0:0.25:1,  titlesize = 22,
    xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    hidespines!(ax, :t, :r)
    # colors = generate_colors(cmap, n_j)
    if size(lines)[2] == length(y_grid)
        grid = y_grid
    elseif size(lines)[2] == length(x_grid)
        grid = x_grid
    else
        error("Lines must have the same length as the grid")
    end
    for j ∈ 1:n_j
        lines!(grid, lines[j, :]; label = "Location $(j)", linewidth = 4, color = colors[j])
        # lines!(y_grid, lines[j, :]; label = "Location $(j)", linewidth = 4)
    end

    axislegend(""; position = :rc, bgcolor = (:grey90, 0.25));

    fig
end
#==========================================================================================
# TODO: Add description
#description: 
==========================================================================================#
function plot_unemployed_search(prim::Primitives, res::Results; cmap = :plasma, backgroundcolor = "#FAFAFA")
    @unpack n_x, x_grid, n_j = prim
    @unpack ϕ_u = res

    fig = Figure(resolution = (600, 400), fonts = (; regular= texfont(), bold = regular= texfont()), 
                backgroundcolor = backgroundcolor)

    ax = Axis(fig[1,1], xlabel = "Worker Type", ylabel = "", title = "",
    ylabelsize = 18, xticks = 0:0.25:1,  titlesize = 22,
    xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    hidespines!(ax, :t, :r)

    # colors  = generate_colors(cmap, n_j)
    for j ∈ 1:n_j
        lines!(ax, x_grid, ϕ_u[j, 1, :]; label = "Location $(j)", linewidth = 4, color = colors[j])
        # lines!(ax, x_grid, ϕ_u[j, 1, :]; label = "Location $(j)", linewidth = 4)
    end

    axislegend(""; position = :rc, bgcolor = (:grey90, 0.25));

    fig

    return fig
end
#==========================================================================================
# TODO: Add description
#description: 
==========================================================================================#
function plot_employed_search(prim::Primitives, res::Results; j_prime::Int= 1,
                            xlabel = "", ylabel ="", title ="", cmap=:plasma, backgroundcolor = "#FAFAFA")
    # Extract necessary variables
    @unpack x_grid, y_grid, n_j = prim

    zmin, zmax = minimum(res.ϕ_s[:, j_prime, :, :]), maximum(res.ϕ_s[:, j_prime, :, :])

    fig = Figure(resolution = (n_j .* 400, 450), fonts = (; regular= texfont(), bold = regular= texfont()), 
                backgroundcolor = backgroundcolor)
    axs = [Axis(fig[1, j], xlabel = xlabel, ylabel = (j == 1) ? ylabel : "",
    ylabelsize = 18,  titlesize = 22,
    xticks = 0:0.25:1, yticks = 0.25:0.25:1, title = "Location $(j)",
    xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0) for j ∈ 1:n_j]

    for j ∈ 1:n_j
        z = res.ϕ_s[j, j_prime, :, :]
        CairoMakie.heatmap!(axs[j], x_grid, y_grid, z, 
        colormap = (cmap,  0.85), clim = (zmin, zmax), 
        colorrange = (zmin, zmax), colorbar = (j == 2), 
        colorbar_orientation = :vertical, colorbar_position = :right)
        CairoMakie.contour!(axs[j],  x_grid, y_grid, z .> 0,; color=:white, levels=1, linewidth=3)
    end
    Label(fig[0, :], text = title, fontsize = 32, halign = :center, valign = :center)
    return fig
end
#*=========================================================================================
#* Plot All
#*=========================================================================================
function plot_all_model(prim::Primitives, res::Results, dist::DistributionsModel; cmap = :plasma, backgroundcolor = "#FAFAFA")
    # Instantiate figures
    figures = Figures()
    # Plot distributions
    plot_equilibrium_distribution(prim, dist, figures, 
                        cmap = cmap, backgroundcolor = backgroundcolor)
    # Plot surviving matches
    plot_surviving_matches(prim, res, figures, title = "",
                        cmap = cmap, backgroundcolor = backgroundcolor)
    # Plot lines
    ## Poaching index
    figures.poaching_rank = plot_lines_model(prim, res.Π, xlabel = "Firm Type", title = "", 
                        cmap = cmap, backgroundcolor = backgroundcolor)
    ## Posted Vacancies
    figures.vacancies = plot_lines_model(prim, res.v, xlabel = "Firm Type", title = "", 
                        cmap = cmap, backgroundcolor = backgroundcolor)
    ## Distribution of Vacancies
    figures.eq_vacancies = plot_lines_model(prim, res.γ, xlabel = "Firm Type", title = "", 
                        cmap = cmap, backgroundcolor = backgroundcolor)
    # emp = dropdims(sum(dist.h, dims=2) , dims=2)
    # figures.eq_vacancies = plot_lines_model(prim, emp, xlabel = "Firm Type", title = "", 
    #                     cmap = cmap, backgroundcolor = backgroundcolor)
    # Compute aggregate distributions
    ℓ_agg = sum(dist.ℓ, dims=1) ./ sum(dist.ℓ)
    u_agg = sum(dist.u, dims=1)[:] ./ sum(dist.u)
    h_agg = sum(dist.h, dims=(1,3))[:] ./ sum(dist.h)

    figures.agg_dist = plot_distribution_2D( prim, vcat(h_agg', u_agg'), [1,1], 
                                            labels = ["All Employed", "Unemployed"],  
                                            cmap = cmap, xlabel = "Worker Type", 
                                            title = "", 
                                            backgroundcolor= backgroundcolor)

    figures.search_unemp = plot_unemployed_search(prim, res, cmap = cmap, backgroundcolor = backgroundcolor)

    figures.search_emp = plot_employed_search(prim, res, cmap = cmap, backgroundcolor = backgroundcolor)
    
    # Return figures
    return figures
end # plot_all_model


