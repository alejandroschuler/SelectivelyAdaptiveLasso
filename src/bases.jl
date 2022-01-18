#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *, intersect
import StatsBase: sample
import Combinatorics: powerset
using InvertedIndices


BasisIndex = Set{Tuple{Int, Float64}}

struct BasisVector
    length::Int
    nzind::Vector{UInt32} 
end

function *(a::Float64, x::BasisVector)::Vector{Float64}
    result = zeros(Float64, x.length)
    result[x.nzind] .= a
    return result
end
*(x::BasisVector, a::Float64) = a*x

function *(v::Vector{Float64}, x::BasisVector)::Float64
    if length(x.nzind) == 0
        return 0.0
    end
    return sum(v[i] for i in x.nzind)
end
*(x::BasisVector, v::Vector{Float64}) = v*x

hash(x::BasisVector) = hash(x.nzind)



struct FeatureVector
    n::Int # number of observations
    sorted::Vector{Float64} # S
    sorted_to_raw::Vector{UInt32} # an index I that makes S[i] = X[I[i]]
    raw_to_sorted::Vector{UInt32} # an index I: S[I[i]] = X[i]
end

function FeatureVector(X::AbstractVector)::FeatureVector
    n = length(X)
    sorted_to_raw = sortperm(X, rev=true)
    raw_to_sorted = sortperm(sorted_to_raw)
    sorted = X[sorted_to_raw]
    return FeatureVector(n, sorted, sorted_to_raw, raw_to_sorted)
end

function findfirst_gte_sorted(S::Vector{Float64}, x::Float64)
    # S should be sorted in reverse
    for (i,xi) in enumerate(S)
        if xi < x return i-1 end
    end
    return length(S)
end

function one_way(X::FeatureVector, x)
    final = findfirst_gte_sorted(X.sorted, x)
    if final == 0 return Vector{UInt32}() end
    return X.sorted_to_raw[1:final]
end

function build_basis(X::Vector{FeatureVector}, idx::BasisIndex)
    return BasisVector(
        X[1].n,
        intersect([
            j==0 ? (1:X[1].n) : one_way(X[j], x) 
            for (j,x) in idx
        ]...)
    )
end

function sortperm_subset(X::FeatureVector, I)
    # given an index I of length b relative to the the unsorted vector X, return the 
    # subset permutation index Ĩ for X so that X[I[Ĩ]] is sorted, 
    # i.e. S[sort(raw_to_sorted[I])] = X[I[Ĩ]]
    return sortperm(X.raw_to_sorted[I])
end

changepoints(x::Vector{Float64}) = [i for i in 1:(length(x)-1) if x[i] ≠ x[i+1]]

function sse_indicator_ols(Y::Vector{Float64}, Y2::Vector{Float64}=Y.^2)
    # [SSE(Y, Ŷₖ) for k∈1:n] where Ŷₖ is the OLS prediction of Y regressed on [1,1,... 1ₖ, 0... 0]
    Y_sums = cumsum(Y)
    Y_sq_sums = cumsum(Y2)
    nnz = 1:length(Y)

    return (Y_sq_sums - (Y_sums.^2) ./ nnz) + (Y_sq_sums[end] .- Y_sq_sums)
    # ρ_abs = abs.(Y_sums)
    # sses = sses./ρ_abs
    # sses = -ρ_abs
end

function best_split_sorted(
    I, # index within unsorted: X[I] is sorted
    X::Vector{Float64},
    Y::Vector{Float64}, 
    Y2::Vector{Float64} = Y.^2,
    off_limits = Set{Float64}(),
)::Tuple{Any, Vector{Float64}, Vector{Float64}, Float64, Float64}
    # to allow for splits on either side (≥ OR <) just do the cumsums in both
    # directions! Just be careful with the valid cutpoints ^
    sses = sse_indicator_ols(Y, Y2)
    sse_order = sortperm(sses)
    # changes = changepoints(X)
    for ĩ in sse_order
        x = X[ĩ]
        # choosing x = min(X[I]) doesn't do anything
        if (x ∉ off_limits) # & (ĩ ∈ changes) 
            return I[1:ĩ], Y[1:ĩ], Y2[1:ĩ], x, sses[ĩ]
        end
        return I, Y, Y2, NaN, Inf
    end
end

function interact_basis(
    I,
    X::Vector{FeatureVector}, 
    Y::Vector{Float64}, 
    Y2::Vector{Float64} = Y.^2,
    features = 1:length(X),
    off_limits::Set{Tuple{Int, Float64}} = Set{Tuple{Int, Float64}}(),
)
    results = []
    for j in features
        Ĩ = sortperm_subset(X[j], I)
        I_sorted = I[Ĩ]
        push!(results, best_split_sorted(
            I_sorted, 
            X[j].sorted[X[j].raw_to_sorted[I_sorted]], 
            Y[Ĩ], 
            Y2[Ĩ], 
            Set(x for (k,x) in off_limits if k==j),
        ))
    end
    Is, Ys, Y2s, xs, sses = zip(results...)
    sse, j = findmin(sses)
    return Is[j], Ys[j], Y2s[j], (features[j], xs[j]), sses[j]
end

filter_and_pare(Ω, ω) = Set(delete!(A,ω) for A in Ω if ω in A) # ω ∈ A ∈ Ω

function basis_search(
    X::Vector{FeatureVector},
    Y::Vector{Float64};
    subsample_n::Int=length(Y), 
    feat_n::Int=Int(ceil(sqrt(length(X)))),
    off_limits = Set([BasisIndex([(0, NaN)])]), # iterable of [BasisIndex]es
)::Tuple{BasisIndex, BasisVector}
    #=
    Greedily attempt to find the basis vector that will be most useful to add to the lasso by interacting 
    one-way basis vectors, one at a time.
    
    This "top-down" approach makes sense heuristically because we expect that low-order interactions are 
    what's important for most real-world data-generating processes. There is also the fact that realizations
    of higher-order sections must have greater and greater sparsity since these are products of h∈{0,1}ⁿ.
    =#  
    n = length(Y)
    I = sample(1:n, subsample_n, replace=false)
    Y = Y[I]
    Y2 = Y.^2

    current_sse = Inf
    basis_index = BasisIndex()
    orig_off_limits = deepcopy(off_limits)
    
    while true
        features = sample(1:length(X), feat_n, replace=false)
        off_limits_coords = Set(first(b) for b in off_limits if length(b)==1)

        Ĩ, Y, Y2, coord, sse = interact_basis(
            I, X, Y, Y2,
            features, 
            off_limits_coords ∪ basis_index
        )
        if (current_sse ≤ sse) | (current_sse ≈ sse) 
            if length(basis_index) == 0
                basis_index = Set{Tuple{Int,Float64}}([(0, NaN)]) 
            end
            if subsample_n == n
                return basis_index, BasisVector(n, I)
            else
                return basis_index, build_basis(X, basis_index)
            end
        else 
            current_sse, I = sse, Ĩ
            push!(basis_index, coord)
            off_limits = filter_and_pare(off_limits, coord)
        end
    end
end

function basis_search_random(X::Vector{FeatureVector})::Tuple{BasisIndex, BasisVector}
    features = []
    n, p = X[1].n, length(X)
    while length(features) == 0
        features = (1:p)[sample([false,true], p)]
    end
    basis_index = Set((j, X[j].sorted[rand(1:n)]) for j in features)
    return basis_index, build_basis(X, basis_index)
end






struct Bases
    dict::Dict{BasisIndex, BasisVector}
    set::Set{BasisVector}
    sum::Dict{BasisIndex, Int}
end

function Bases(
    X::Vector{FeatureVector}; 
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

function *(X::Bases, β::Dict{BasisIndex, Float64})::Vector{Float64}
    total = zeros(Float64, X.sum[BasisIndex([(0,NaN)])])
    for (b, βi) in β
        # total[X[b].nzind] .+= βi
        for i in X[b].nzind
            total[i] += βi
        end
    end
    return total
end

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

function delete_basis!(
    bases::Bases, 
    index::BasisIndex
)::Bases
    if index in keys(bases.dict)
        delete!(bases.set, bases.dict[index])
        delete!(bases.dict, index)
        delete!(bases.sum, index)
    end
    return bases
end

function filter_bases!(bases::Bases, indices)
    indices = push!(Set(indices), Set([(0.0, NaN)]))
    for b in setdiff(Set(keys(bases)), Set(indices))
        delete_basis!(bases, b)
    end
    return bases
end
