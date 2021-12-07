#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *
import StatsBase: sample
import Combinatorics: powerset

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

function basis_search(
        bases::Bases,
        Y::Vector{Float64};
        n_subsample::Int = 1000,
        m_subsample::Int = 1000,
    )::Tuple{BasisIndex, BitVector}
    #=
    Attempt to find the basis that will be most useful to linearly predict Y.
    
    Greedily searches through sections, starting at one-way. Starting with the intercept, this function creates all 
    of the interactions between a basis and all one-way bases. It then searches through all of the 
    bases within the m*p newly-created candidate bases and finds the basis with the maximum univariate 
    regression coefficient on the outcome. If this is greater than that of the previous iteration, it 
    replaces the current basis with the the found basis. Otherwise it returns the current basis.
    
    This "top-down" approach makes sense heuristically because we expect that low-order interactions are 
    what's important for most real-world data-generating processes. There is also the fact that realizations
    of higher-order sections must have greater and greater sparsity since these are products of h∈{0,1}ⁿ.
    =#     
    n_subsample = min(n_subsample, bases.n)
    m_subsample = min(m_subsample, bases.m)
    obs_subsample_idx = sample(1:bases.n, n_subsample, replace=false)
    knots_subsample_idx = sample(1:bases.m, m_subsample, replace=false)

    Y_subsample = Y[obs_subsample_idx]
    one_way_bases = bases.one_way[obs_subsample_idx, knots_subsample_idx, :]
    candidate_bases = copy(one_way_bases)

    basis_index = BasisIndex([CartesianIndex(0,0)]) # start with intercept
    basis = BitVector(ones(n_subsample))
    max_β = Y_subsample' * basis/n_subsample # the current strength of the intercept
    
    while true
        β = abs.(
            sum(Y_subsample .* candidate_bases, dims=1) ./
            sum(candidate_bases, dims=1)
        )
        β[isnan.(β)] .= -Inf
        new_max_β, idx = findmax(β)
        _, knot, one_way_section = Tuple(idx)

        if new_max_β ≤ max_β # no improvement, return last section and bases
            # also catches the terminal case where we get the same thing twice in a row
            return basis_index, basis
        else
            push!(basis_index, CartesianIndex(knot, one_way_section))
            basis = candidate_bases[:, knot, one_way_section]
            candidate_bases .= basis .* one_way_bases
            max_β = new_max_β
        end
    end
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

    bases = Bases(
        knots, 
        # sections,
        one_way, dict, set, sums, 
        n, m, p
    )

    return bases
end

function *(X::Bases, β::Dict{BasisIndex, Float64})
    return sum(X[b]*β[b] for b in keys(β))
end

function build_basis(bases::Bases, index::BasisIndex)::BitVector
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



