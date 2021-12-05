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
    # Functional properties
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
    Attempt to find the section that will be most useful to linearly predict Y.
    
    Greedily searches through sections, starting at one-way. Given a k-way section, this function creates all 
    of the interactions between that section and all one-way sections. It then searches through all of the 
    bases within the p newly-created candidate sections and finds the basis with the maximum univariate 
    regression coefficient on the outcome. If this is greater than that of the previous iteration, it 
    replaces the k-waysection with the (k+1)-way section corresponding to that found basis. If the chosen section
    is already included in the basis set, we continue further down the "tree" of interactions.
    
    This "top-down" approach makes sense heuristically because we expect that low-order interactions are 
    what's important for most real-world data-generating processes. There is also the fact that realizations
    of higher-order sections must have greater and greater sparsity since these are products of h∈{0,1}ⁿ.
    
    I also tried choosing the section using the sum of the univariate coefficients across all bases within 
    each candidate section. This is slower because of the sum and doesn't appear to increase 
    "performance". It selects much deeper bases in general; perhaps worth revisting later. I also tried a 
    recursive implementation but looping is faster.
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
            # also catches the terminal case where a single one-way section is added twice in a row
            return basis_index, basis
        else
            push!(basis_index, CartesianIndex(knot, one_way_section))
            basis = candidate_bases[:, knot, one_way_section]
            candidate_bases .= basis .* one_way_bases
            max_β = new_max_β
        end
    end
end

# function add_section!(
#     bases::Bases, 
#     section::Section, 
#     section_bases::Union{BitMatrix,Nothing}=nothing
# )::Bases
#     # Adds the given sectional bases to the bases object, deduplicating anything that already exists

#     if section == Section()
#         return bases
#     end

#     if isnothing(section_bases)
#         section_bases = prod(bases.one_way[:,:,[section...]], dims=3)[:,:,1]
#     end

#     n, m = size(section_bases)
#     for (knot, basis) in enumerate(eachcol(section_bases))
#         if basis ∉ bases.set
#             b = (knot, section)
#             push!(bases.set, basis)
#             bases.dict[b] = basis
#             bases.sum[b] = sum(basis)
#             push!(bases.sections, section)
#         end
#     end
#     return bases
# end

# function expand!(bases::Bases, Y::Vector{Float64})::Tuple{Bases, Section}
#     #=
#     Add a section to the set of bases. Uses a heuristic search to find the section most useful to predict Y. If said 
#     section is already included, pick the first section (in increasing order of interactions) that isn't. This is 
#     unlikely to happen since sections that have already been picked are residualized out.
#     =#
#     section, section_bases = section_search(bases, Y)
#     if section in bases.sections # already included, pick something random
#         n,m,p = size(bases.one_way)
#         for section in (Section(s) for s in powerset(1:p))
#             if section ∉ bases.sections
#                 return add_section!(bases, section), section
#             end
#         end
#     end
#     return add_section!(bases, section, section_bases), section
# end

function Bases(
    X::Matrix{Float64};
    knots::Matrix{Float64} = X,
    # sections::Set{Section} = Set{Section}()
    # init_one_way=false
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
    # for section in sections
    #     add_section!(bases, section)
    # end

    return bases
end

function *(X::Bases, β::Dict{BasisIndex, Float64})
    return sum(X[b]*β[b] for b in keys(β))
end

function add_basis!(
    bases::Bases, 
    index::BasisIndex; 
    basis::Union{Nothing,BitVector}=nothing
)::Bases
    if isnothing(basis)
        basis = prod(bases.one_way[:,[(i for i in index if i ≠ CartesianIndex(0,0))...]], dims=2)[:,1]
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
        add_basis!(bases, b)
    end
    return bases.dict[b]
end



