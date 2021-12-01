module SelectivelyAdaptiveLasso

# temp for dev
export Bases, expand!, add_section!, section_search
export coordinate_descent
# eventual exports
export SALSpec, SALFit, fit, predict

include("bases.jl")
include("ccd.jl")

import StatsBase: sample
import Statistics: mean

Section = Set{Int}

mutable struct SALSpec
    # stopping hyperparameters
    λ::Float64 # minimum desired regularization strength 
    B::Int # maximum number of sections to adaptively select and use
    max_iter::Int # maximum iterations to step through
    
    # fitting hyperparameters
    sections::Set{Section} # sections to always include 
    λ_ratio::Float64 # multiplicative increment between regularization: λₖ₊₁ = rλₖ 
    m_knots::Int # number of data points from training data to use as knots
    n_subsample::Int # number of observations to use in section selection
    m_subsample::Int # number of knots to use in section selection
end

function SALSpec(;
        λ::Real = 0,
        B::Int = typemax(Int),
        max_iter::Int = typemax(Int),
        sections::Set{Section} = Set([Section()]),
        λ_ratio::Real = 2/3,
        m_knots::Int = typemax(Int), # let the data choose
        n_subsample::Int = 500,
        m_subsample::Int = 500,
    )
    if (λ==0) & B==typemax(Int) & max_iter==typemax(Int)
        error("One of λ, B, or max_iter must be provided")
    end

    if (λ < 0) | (1 ≤ λ)
        error("requires λ ∈ [0,1)")
    end        
    
    if (λ_ratio ≤ 0) | (1 ≤ λ_ratio)
        error("requires λ_ratio ∈ (0,1)")
    end
    
    return SALSpec(
        float(λ), B, max_iter, 
        sections, λ_ratio,
        m_knots, n_subsample, m_subsample
    )
end

mutable struct SALFit
    knots::Matrix{Float64} # (m x p) matrix, each row is a knot
    sections::Set{Section}
    β::Dict{Tuple{Int, Section}, Float64} # maps (knot, section) => coefficient
end

function predict(sal_fit::SALFit, X::Matrix{Float64})
    bases = Bases(X, knots=sal_fit.knots, sections=copy(sal_fit.sections))
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
    )::Tuple{SALFit, Any}
    #= 
    Fits the selectively adaptive lasso
    =#
    if !isnothing(X_val) & !isnothing(Y_val)
        val = true
    end
    
    n,p = size(X)
    m = min(n, sal.m_knots)
    knots = X[sample(1:n, m, replace=false),:]
    bases = Bases(X, knots=knots, sections=copy(sal.sections))
    λ = 1.0 # pick initial λ so that |Yᵢ| ≤ max(Y) => β=0
    λ_scale = n*max(Y...)
    
    β, R, _ = coordinate_descent(bases, Y, λ=λ*λ_scale)
    loss = [mean(R.^2)] 
    λ_record = [λ]
    section_record = [Section()]
    if val
        bases_val = Bases(X_val, knots=bases.knots, sections=copy(bases.sections))
        R_val = Y_val - bases_val*β
        loss_val = [mean(R_val.^2)]
    end
    
    for i in 0:sal.max_iter
        if (λ < sal.λ) | (length(bases.sections) ≥ sal.B)
            break
        end
        
        section = section_search(
            bases, R, 
            n_subsample=sal.n_subsample, 
            m_subsample=sal.m_subsample,
        )
        if section in bases.sections
            λ *= sal.λ_ratio
        else
            add_section!(bases, section)
        end
        
        β, R, l = coordinate_descent(bases, Y, λ=λ*λ_scale, β=β)
        push!(loss, mean(R.^2))
        if val
            R_val = Y_val - bases_val*β
            push!(loss_val, mean(R_val.^2))
        end
        push!(λ_record, λ)
        push!(section_record, section)

        print((loss[end], loss_val[end], λ, section))
        print("\n")
    end

    return SALFit(bases.knots, bases.sections, β), (loss, loss_val, λ_record, section_record)
end

end
