#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *
import StatsBase: sample
import Combinatorics: powerset
using InvertedIndices

BasisIndex = Set{CartesianIndex{2}}
function section(index::BasisIndex)::Set{Int}
    Set([one_way_index[2] for one_way_index in index])
end

mutable struct Bases
    #=
    Represents bases of the form 

    [h(Xᵢ): i ∈ 1...n]
    where h(x) = Πⱼ1(xⱼ ≥ cⱼ) 
    and cⱼ ∈ {Xᵢⱼ: i ∈ 1...n}

    We call these "SAL bases" (the mappings h) or "empirical SAL bases" (the vectors h(X)). 
    SAL bases are a superset of the original HAL bases defined by knots {Xᵢ : i}. 
    They are also equivalent to indicators for all possible leaves in a standard regression tree.

    Note that we call any basis of the specific form h(x) = 1(xⱼ ≥ c) a "one-way" basis.
    =#
    knots::Matrix{Float64}

    # Realizational properties
    one_way::BitArray{3}
    dict::Dict{BasisIndex, BitVector}
    set::Set{BitVector}
    sum::Dict{BasisIndex, Int}
    n::Int # number of observations
    m::Int # number of knots
    p::Int # dimensionality of Xᵢ
    X_sort_idx::Vector{Vector{Int}}
    X_sort_idx_reverse::Vector{Vector{Int}}
end

function build_one_way(
        X::Matrix{Float64}, # (n x p)
        knots::Matrix{Float64} = X, # optional, data points to use as knots
    )::BitArray{3}
    #=
    Helper for Bases constructor.

    Creates all bases corresponding to one-way sections, using the knots provided.
    Returns an (n x m x p) BitArray.
    =#

    n, p = size(X)
    m, p = size(knots)

    one_way = zeros(Bool,(n,m,p))

    for knot in 1:m
        one_way[:,knot,:] = (X' .≥ knots[knot, :])'
    end

    return BitArray(one_way) # (n x m x p), usually m = n
end

function BitMatrix(bases::Bases)::BitMatrix
    BitMatrix(hcat([basis for basis in values(bases.dict)]...))
end

function index_to_bool(int_index::Vector{Int}, n::Int)
    # Translates a 1D integer index to a boolean index vector. Helper function for interact_basis.
    bool_index = Vector{Bool}(zeros(n))
    bool_index[int_index] .= 1
    return bool_index
end

function interact_basis(Y::Vector{Float64}, bases::Bases, λ, Y_sum_squares, basis_ints, features)
    best_metric = Inf
    next_split = ([0], 0)
    if length(basis_ints) == 0
        return next_split, best_metric
    end

    for j in features
        idx_idx = index_to_bool(bases.X_sort_idx_reverse[j][basis_ints], bases.n)
        idx = @view bases.X_sort_idx[j][idx_idx]
        for i in 1:length(idx)
            new_basis_ints = @view idx[i:end]
            Y_basis = @view Y[new_basis_ints] # {Yᵢ : h(Xᵢ)=1}
            sse = sum((Y_basis .- mean(Y_basis)).^2) + (Y_sum_squares - sum(Y_basis.^2)) # ∑(Y-βh(X))²
            ρ_abs = abs(sum(Y_basis)) # ρ = Y'h(X). Note β = softmax(ρ, λ)/∑h(X) ⟹ β=0 if |ρ|≤λ
            metric = sse/ρ_abs # want both small sse and big ρ
            if (ρ_abs > λ) & (metric < best_metric)
                best_metric = metric
                next_split = Tuple((new_basis_ints, j))
            end
        end
    end
    
    return next_split, best_metric
end

function basis_search(
    bases::Bases,
    Y::Vector{Float64},
    λ::Float64;
    subsample_n::Int=bases.n, 
    feat_n::Int=bases.p,
)::BasisIndex
    #=
    Attempt to find the basis that will be most useful to linearly predict Y.
    
    Greedily searches through sections. Starting with the intercept, this function creates all 
    of the interactions between a basis and all one-way bases. It then searches through all of the 
    bases within the m*p newly-created candidate bases and finds the basis with the maximum dot product with the
    outcome (> λ guarantees an update in the lasso CCD algorithm).
    If this is greater than that of the previous iteration, it 
    replaces the current basis with the the found basis. Otherwise it returns the current basis.
    
    This "top-down" approach makes sense heuristically because we expect that low-order interactions are 
    what's important for most real-world data-generating processes. There is also the fact that realizations
    of higher-order sections must have greater and greater sparsity since these are products of h∈{0,1}ⁿ.
    =#  
    basis_ints = sample(1:bases.n, subsample_n, replace=false)
    
    best_metric = Inf
    basis_index = BasisIndex([CartesianIndex(0,0)]) # start with intercept
    Y_sum_squares = sum(Y[basis_ints].^2)
    
    while true
        features = sample(1:bases.p, feat_n, replace=false)
        (basis_ints, feat), metric = interact_basis(
            Y, bases, λ* subsample_n/bases.n, 
            Y_sum_squares, basis_ints, features
        )
        if metric < best_metric
            best_metric = metric
            push!(basis_index, CartesianIndex(basis_ints[1], feat))
        else
            return basis_index
        end
    end
    
end

function basis_search_random(bases::Bases)::BasisIndex
    features = []
    while length(features) == 0
        features = (1:bases.p)[sample([false,true], bases.p)]
    end
    knots = sample(1:bases.m, length(features))
    return Set(CartesianIndex(knot, feat) for (knot, feat) in zip(knots, features))
end

function Bases(
    X::Matrix{Float64};
    knots::Matrix{Float64} = X,
)::Bases
    one_way = build_one_way(X, knots)
    n,m,p = size(one_way)

    intercept = BitVector(ones(n))

    # add intercept
    set = Set{BitVector}([intercept])
    dict = Dict{BasisIndex, BitVector}(BasisIndex([CartesianIndex(0,0)]) => intercept)
    sums = Dict{BasisIndex, Int}(BasisIndex([CartesianIndex(0,0)]) => n)

    X_sort_idx = [sortperm(col) for col in eachcol(X)] 
    X_sort_idx_reverse = [sortperm(i) for i in X_sort_idx]

    bases = Bases(
        knots, 
        one_way, dict, set, sums, 
        n, m, p,
        X_sort_idx, X_sort_idx_reverse,
    )

    return bases
end

function *(X::Bases, β::Dict{BasisIndex, Float64})
    return sum(X[b]*β[b] for b in keys(β))
end

function build_basis(bases::Bases, index::BasisIndex)::BitVector
    if index == BasisIndex([CartesianIndex(0,0)]) 
        return bases[BasisIndex([CartesianIndex(0,0)])]
    end
    clean_index = (i for i in index if i ≠ CartesianIndex(0,0))
    return prod(
        bases.one_way[:,[clean_index...]], 
        dims=2
    )[:,1]
end

function add_basis!(
    bases::Bases, 
    index::BasisIndex; 
    basis::Union{Nothing,BitVector}=nothing
)::Bases
    if isnothing(basis)
        basis = build_basis(bases, index)
    end
    s = sum(basis)
    if s ≠ 0 # don't add a null basis
        push!(bases.set, basis)
        bases.dict[index] = basis
        bases.sum[index] = s
    end

    return bases
end

keys(bases::Bases) = keys(bases.dict)

function getindex(bases::Bases, b::BasisIndex)::BitVector
    if b ∉ keys(bases.dict)
        basis = build_basis(bases, b)
        add_basis!(bases, b, basis=basis) # protects from null basis
    else
        basis = bases.dict[b]
    end
    return basis
end



