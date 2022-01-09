#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *, intersect
import StatsBase: sample
import Combinatorics: powerset
using InvertedIndices
using SparseArrays


BasisIndex = Set{Tuple{Int, Float64}}

struct BasisVector
    length::Int
    nzind::Vector{UInt32} 
end

*(a::Float64, x::BasisVector)::Vector{Float64} = sparsevec(x.nzind, a, x.length)
*(x::BasisVector, a::Float64) = a*x

function *(v::Vector{Float64}, x::BasisVector)::Float64
    return sum(v[i] for i in x.nzind)
end
*(x::BasisVector, v::Vector{Float64}) = v*x

hash(x::BasisVector) = hash(x.nzind)



struct Features
    n::Int # number of observations
    p::Int # dimensionality of Xᵢ
    sorted::Vector{Vector{Float64}} # columns of X, sorted
    sort_idx::Vector{Vector{UInt32}} # for each feature p, an index i that makes X[i,p] sorted
    idx_in_sorted::Vector{Vector{UInt32}} # given Xᵢ with position i, this index tells us the 
                                             # position i' in the sorted vector sort(X)
end
function Features(X::Matrix{Float64})::Features
    n,p = size(X)
    sort_idx = [Vector{UInt32}(sortperm(col)) for col in eachcol(X)]
    sorted = [X[i,j] for (j,i) in enumerate(sort_idx)]
    idx_in_sorted = [Vector{UInt32}(sortperm(i)) for i in sort_idx]
    return Features(n, p, sorted, sort_idx, idx_in_sorted)
end

function findfirst_gt_sorted(X_sorted::Vector{Float64}, x::Float64)
    for (i,xi) in enumerate(X_sorted)
        if xi ≥ x return i end
    end
end

function one_way(X::Features, j, x)
    if isnan(x) 
        return 1:X.n
    end
    return X.sort_idx[j][findfirst_gt_sorted(X.sorted[j], x):end]
end

function build_basis(X::Features, idx::BasisIndex)
    return BasisVector(
        X.n,
        intersect([one_way(X,j,x) for (j,x) in idx]...)
    )
end

function interact_basis(
    Y::Vector{Float64}, Y_squared::Vector{Float64}, 
    X::Features, λ::Float64, Y_full_sum_squares::Float64, 
    basis_ints, features::Vector{Int}
)
    best_basis_ints = basis_ints
    best_metric = Inf
    feature, x_split = nothing, nothing

    if length(basis_ints) ≠ 0
        for j in features
            idx_idx = sort(X.idx_in_sorted[j][basis_ints]) 
            idx = @view X.sort_idx[j][idx_idx]

            Y_idx = Y[idx]
            Y_sq_idx = Y_squared[idx]

            Y_sums = cumsum(Y_idx)
            Y_means = Y_sums ./ (1:length(idx))
            Y_sq_sums = cumsum(Y_sq_idx)

            sses = (Y_sq_sums - Y_sums.^2) - (Y_full_sum_squares .- Y_sq_sums)
            ρ_abss = abs.(Y_sums)
            metrics = sses./ρ_abss

            metric, idx_split = findmin(metrics)
            if (ρ_abss[idx_split] > λ) & (metric < best_metric)
                best_basis_ints = idx[idx_split:end]
                best_metric = metric
                feature = j
                x_split = X.sorted[j][X.idx_in_sorted[j][idx[idx_split]]]
            end
        end
    end

    return best_basis_ints, (feature, x_split), best_metric
end

function basis_search(
    X::Features,
    Y::Vector{Float64},
    λ::Float64;
    subsample_n::Int=X.n, 
    feat_n::Int=X.p,
)::Tuple{BasisIndex, BasisVector}
    #=
    Greedily attempt to find the basis vector that will be most useful to add to the lasso by interacting 
    one-way basis vectors, one at a time.
    
    This "top-down" approach makes sense heuristically because we expect that low-order interactions are 
    what's important for most real-world data-generating processes. There is also the fact that realizations
    of higher-order sections must have greater and greater sparsity since these are products of h∈{0,1}ⁿ.
    =#  
    basis_ints = sample(1:X.n, subsample_n, replace=false)
    best_metric = Inf
    basis_index = Set{Tuple{Int,Float64}}([(0, NaN)]) 
    Y_sum_squares = sum(Y[basis_ints].^2)
    Y_squared = Y.^2
    
    while true
        features = sample(1:X.p, feat_n, replace=false)
        new_basis_ints, added_basis_coord, metric = interact_basis(
            Y, Y_squared, X, λ * subsample_n/X.n, 
            Y_sum_squares, basis_ints, features
        )
        if metric < best_metric
            best_metric = metric
            basis_ints = new_basis_ints
            push!(basis_index, added_basis_coord)
        else
            if subsample_n == X.n
                return basis_index, BasisVector(X.n, basis_ints)
            else
                return basis_index, build_basis(X, basis_index)
            end
        end
    end
end

function basis_search_random(X::Features)::Tuple{BasisIndex, BasisVector}
    features = []
    while length(features) == 0
        features = (1:X.p)[sample([false,true], X.p)]
    end
    knots = sample(1:X.n, length(features))
    basis_index = Set((j, X.sorted[j][i]) for (i,j) in zip(knots, features))
    return basis_index, build_basis(X, basis_index)
end






struct Bases
    dict::Dict{BasisIndex, BasisVector}
    set::Set{BasisVector}
    sum::Dict{BasisIndex, Int}
end

function Bases(
    X::Features; 
    indices = Set{BasisIndex}([BasisIndex([(0,NaN)])])
)::Bases
    bases = Bases(
        Dict{BasisIndex, BasisVector}(),
        Set{BasisVector}(),
        Dict{BasisIndex, Int}(),
    )
    for idx in indices
        add_basis!(bases, idx, build_basis(X, idx)) 
    end
    return bases
end

keys(bases::Bases) = keys(bases.dict)
getindex(bases::Bases, b::BasisIndex)::BasisVector = bases.dict[b]

*(X::Bases, β::Dict{BasisIndex, Float64})::Vector{Float64} = sum(X[b]*β[b] for b in keys(β))

function add_basis!(
    bases::Bases, 
    index::BasisIndex,
    basis::BasisVector,
)::Bases
    s = basis.length
    if s ≠ 0 # don't add a null basis
        push!(bases.set, basis)
        bases.dict[index] = basis
        bases.sum[index] = s
    end
    return bases
end

