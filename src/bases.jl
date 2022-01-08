#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *, intersect, hash
import StatsBase: sample
import Combinatorics: powerset
using InvertedIndices

struct BoolSparseVector
    length::Int
    nzind::Vector{Int} 
end
BoolSparseVector(x::BitVector) = BoolSparseVector(length(x), findall(!iszero, x))

function *(a::Float64, x::BoolSparseVector)::Vector{Float64}
    result = zeros(Float64, x.length)
    result[x.nzind] .= a
    return result
end
*(x::BoolSparseVector, a::Float64) = a*x

function *(v::Vector{Float64}, x::BoolSparseVector)::Float64
    return sum(v[i] for i in x.nzind)
end
*(x::BoolSparseVector, v::Vector{Float64}) = v*x

function intersect(one_way_bases::Vararg{BoolSparseVector}) 
    return BoolSparseVector(
        one_way_bases[1].length,
        intersect([b.nzind for b in one_way_bases]...)
    )
end

hash(x::BoolSparseVector) = hash(x.nzind)
getindex(x, i::BoolSparseVector) = x[i.nzind]
# setindex!(x, v, i::BoolSparseVector) = setindex!(x, v, i.nzind)

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
    dict::Dict{BasisIndex, BoolSparseVector}
    set::Set{BoolSparseVector}
    sum::Dict{BasisIndex, Int}
    n::Int # number of observations
    p::Int # dimensionality of Xᵢ
    X_sort_idx::Vector{Vector{Int}}
    X_sort_idx_reverse::Vector{Vector{Int}}
end

function interact_basis(
    Y::Vector{Float64}, Y_squared::Vector{Float64}, 
    bases::Bases, λ::Float64, Y_full_sum_squares::Float64, 
    basis_ints, features::Vector{Int}
)
    best_metric = Inf
    next_split = ([0], 0)
    if length(basis_ints) == 0
        return next_split, best_metric
    end

    for j in features
        idx_idx = sort(bases.X_sort_idx_reverse[j][basis_ints])
        idx = @view bases.X_sort_idx[j][idx_idx]

        Y_idx = Y[idx]
        Y_sq_idx = Y_squared[idx]

        # need to account for possible ties here, i.e. can't break in the middle of a tie
        # should give speedup in exchange for storing another index or something?
        Y_sums = cumsum(Y_idx)
        Y_means = Y_sums ./ (1:length(idx))
        Y_sq_sums = cumsum(Y_sq_idx)

        sses = (Y_sq_sums - Y_sums.^2) - (Y_full_sum_squares .- Y_sq_sums)
        ρ_abss = abs.(Y_sums)
        metrics = sses./ρ_abss

        metric, split = findmin(metrics)
        if (ρ_abss[split] > λ) & (metric < best_metric)
            best_metric = metric
            next_split = Tuple((idx[split:end], j))
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
    Y_squared = Y.^2
    
    while true
        features = sample(1:bases.p, feat_n, replace=false)
        (basis_ints, feat), metric = interact_basis(
            Y, Y_squared, bases, λ* subsample_n/bases.n, 
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
    knots = sample(1:bases.n, length(features))
    return Set(CartesianIndex(knot, feat) for (knot, feat) in zip(knots, features))
end

function Bases(
    X::Matrix{Float64};
)::Bases
    n,p = size(X)

    intercept = BoolSparseVector(n, collect(1:n))
    set = Set{BoolSparseVector}([intercept])
    dict = Dict{BasisIndex, BoolSparseVector}(BasisIndex([CartesianIndex(0,0)]) => intercept)
    sums = Dict{BasisIndex, Int}(BasisIndex([CartesianIndex(0,0)]) => n)

    X_sort_idx = [sortperm(col) for col in eachcol(X)] 
    X_sort_idx_reverse = [sortperm(i) for i in X_sort_idx]

    return Bases(
        dict, set, sums, 
        n, p,
        X_sort_idx, X_sort_idx_reverse,
    )
end

function export_basis(X::Matrix{Float64}, bases::Bases, idx::BasisIndex)
    clean_index = (i for i in idx if i ≠ CartesianIndex(0,0))
    return [(X[i], i[2]) for i in clean_index]
end

function translate_basis(X::Matrix{Float64}, idx)::BoolSparseVector
    return BoolSparseVector(reduce(.*, (X[:,j].≥x for (x,j) in idx)))
end

function *(X::Bases, β::Dict{BasisIndex, Float64})
    return sum(X[b]*β[b] for b in keys(β))
end

function one_way(bases::Bases, knot_id::Int, feat_id::Int)
    return BoolSparseVector(bases.n, bases.X_sort_idx[feat_id][knot_id:end])
end

function build_basis(bases::Bases, index::BasisIndex)::BoolSparseVector
    if index == BasisIndex([CartesianIndex(0,0)]) 
        return bases[BasisIndex([CartesianIndex(0,0)])]
    end
    clean_index = (i for i in index if i ≠ CartesianIndex(0,0))
    return intersect([one_way(bases, Tuple(i)...) for i in clean_index]...)
end

function add_basis!(
    bases::Bases, 
    index::BasisIndex; 
    basis::Union{Nothing,BoolSparseVector}=nothing
)::Bases
    if isnothing(basis)
        basis = build_basis(bases, index)
    end
    s = length(basis.nzind)
    if s ≠ 0 # don't add a null basis
        push!(bases.set, basis)
        bases.dict[index] = basis
        bases.sum[index] = s
    end

    return bases
end

keys(bases::Bases) = keys(bases.dict)

function getindex(bases::Bases, b::BasisIndex)::BoolSparseVector
    if b ∉ keys(bases.dict)
        basis = build_basis(bases, b)
        add_basis!(bases, b, basis=basis) # protects from null basis
    else
        basis = bases.dict[b]
    end
    return basis
end



