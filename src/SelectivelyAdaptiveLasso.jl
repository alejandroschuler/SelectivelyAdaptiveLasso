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
    subsample_n::Union{Int, Nothing}
    feat_pct::Float64
    tol::Float64 # tolerance for lasso coordinate descent convergence (loss = mse + penalty)
end

function SALSpec(;
        λ::Real = 0.01,
        max_iter::Int = 5000,
        bases_per_iter::Int = 1,
        subsample_pct::Float64 = 0.1,
        subsample_n::Union{Int, Nothing} = nothing,
        feat_pct::Float64 = 1,
        tol::Real = 1e-4,
    )
    return SALSpec(
        float(λ), max_iter, bases_per_iter,
        subsample_pct, subsample_n, feat_pct,
        float(tol),
    )
end

mutable struct SALFit
    β::Dict{BasisIndex, Float64} 
end

function predict(sal_fit::SALFit, X::Matrix{Float64})
    bases = Bases(
        Features([FeatureVector(x) for x in eachcol(X)]), 
        indices=keys(sal_fit.β)
    )
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
    # logging
        verbose::Bool = false,
        print_iter::Int = 100,
    )::Tuple{SALFit, Any}
    #= 
    Fits the selectively adaptive lasso.
    =#

    val = (!isnothing(X_val) & !isnothing(Y_val))
    n,p = size(X)
    X = [FeatureVector(x) for x in eachcol(X)]
    λ = sal.λ * max(sum(Y[Y.>0]), sum(Y[Y.<0])) # makes it so λ=1 => β=0
    ρ = 0

	bases = Bases(X)
	β, R, _ = coordinate_descent(bases, Y, λ=λ, tol=sal.tol)

    mse = [mean(R.^2)] 
    if val
        X_val = [FeatureVector(x) for x in eachcol(X_val)]
        bases_val = Bases(X_val)
        R_val = Y_val - bases_val*β
        mse_val = [mean(R_val.^2)]
    end

    if isnothing(sal.subsample_n)
        subsample_n = min(Int(ceil(sal.subsample_pct*n)), n)
    else
        subsample_n = sal.subsample_n
    end
    feat_n = min(Int(ceil(sal.feat_pct*p)), p)
    
    for i in 1:sal.max_iter

    	for j in 1:sal.bases_per_iter
            index, basis = basis_search(
                X, R, 
                subsample_n=subsample_n, feat_n=feat_n,
                off_limits = deepcopy(keys(bases))
            )
	        add_basis!(bases, index, basis)
            if val
                basis_val = build_basis(X_val, index)
                add_basis!(bases_val, index, basis_val)
            end
	    end
        
        β, R, l = coordinate_descent(bases, Y, λ=λ, β=β, tol=sal.tol)
        filter_bases!(bases, keys(β))
        
        push!(mse, mean(R.^2))
        if val
            R_val = Y_val - bases_val*β
            push!(mse_val, mean(R_val.^2))
        end

        if verbose & (i % print_iter == 0)
		    print((sqrt(mse[end]), sqrt(mse_val[end]), length(β), length(bases.dict)))
		    print("\n")
		end
    end

    return SALFit(β), (mse, mse_val)
end

end # module
