module SelectivelyAdaptiveLasso

# temp for dev
export Bases, basis_search
export coordinate_descent
# eventual exports
export SALSpec, SALFit, fit, predict

include("bases.jl")
include("ccd.jl")

import StatsBase: sample
import Statistics: mean

mutable struct SALSpec
    # stopping hyperparameters
    λ::Float64 # desired regularization strength
    max_iter::Int # maximum iterations to step through
    
    # fitting hyperparameters
    bases_per_iter::Int # number of bases to add at once
    subsample_pct::Float64
    feat_pct::Float64
    tol::Float64 # tolerance for lasso coordinate descent convergence (loss)
end

function SALSpec(;
        λ::Real = 0.01,
        max_iter::Int = 5000,
        bases_per_iter::Int = 1,
        m_knots::Int = typemax(Int), # let the data choose
        subsample_pct::Float64 = 0.1,
        feat_pct::Float64 = 1,
        tol::Real = 1e-4,
    )
    return SALSpec(
        float(λ), max_iter, bases_per_iter,
        subsample_pct, feat_pct,
        float(tol),
    )
end

mutable struct SALFit
    knots::Matrix{Float64} # (m x p) matrix, each row is a knot
    β::Dict{BasisIndex, Float64} 
end

function predict(sal_fit::SALFit, X::Matrix{Float64})
    bases = Bases(X, knots=sal_fit.knots)
    return bases * sal_fit.β
end

function fit(
    # model object and hyperparameters
        sal::SALSpec,
    # training data
        X::Matrix{Float64}, 
        Y::Vector{Float64};
    # validation data for early stopping
        X_val::Union{Nothing, Matrix{Float64}} = nothing, 
        Y_val::Union{Nothing, Vector{Float64}} = nothing,
    # existing fit for warm-starting
        sal_fit::Union{Nothing, SALFit} = nothing,
    # logging
        verbose::Bool = false,
        print_iter::Int = 100,
    )::Tuple{SALFit, Any}
    #= 
    Fits the selectively adaptive lasso.
    =#

    val = (!isnothing(X_val) & !isnothing(Y_val))
    n,p = size(X)
    λ = sal.λ * max(sum(Y[Y.>0]), sum(Y[Y.<0])) # makes it so λ=1 => β=0
    ρ = 0

    if isnothing(sal_fit)
	    bases = Bases(X)
	    β, R, _ = coordinate_descent(bases, Y, λ=λ, tol=sal.tol)
	else
		bases = Bases(X, knots=sal_fit.knots)
		β = sal_fit.β
		R = Y - bases*β
	end

    loss = [mean(R.^2)] 
    if val
        bases_val = Bases(X_val, knots=bases.knots)
        R_val = Y_val - bases_val*β
        loss_val = [mean(R_val.^2)]
    end
    
    for i in 1:sal.max_iter

        # max_ρ = max(sum(R[R.>0]), sum(R[R.<0]))

    	for j in 1:sal.bases_per_iter
            index = basis_search(bases, R, λ, subsample_pct=sal.subsample_pct, feat_pct=sal.feat_pct)
            basis = build_basis(bases, index)
            ρ = abs(sum(Y[basis]))
            tries = 0 
	        while (index in keys(bases)) | (ρ ≤ λ)
                index = basis_search_random(bases)
                basis = build_basis(bases, index)
                ρ = abs(sum(Y[basis]))
                tries +=1 
                if tries > 1e3
                    return SALFit(bases.knots, β), (loss, loss_val)
                end
	        end
	        add_basis!(bases, index)
	    end
        
        β, R, l = coordinate_descent(bases, Y, λ=λ, β=β, tol=sal.tol)
        push!(loss, mean(R.^2))
        if val
            R_val = Y_val - bases_val*β
            push!(loss_val, mean(R_val.^2))
        end

        if verbose & (i % print_iter == 0)
		    print((loss[end], loss_val[end]))
		    print("\n")
		end
    end

    return SALFit(bases.knots, β), (loss, loss_val)
end

end # module
