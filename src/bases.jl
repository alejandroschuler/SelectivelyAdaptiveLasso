#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix, *
import StatsBase: sample
import Combinatorics: powerset

Section = Set{Int}

mutable struct Bases
    # Functional properties
    knots::Matrix{Float64}
    sections::Set{Section}

    # Realizational properties
    one_way::BitArray{3}
    dict::Dict{Tuple{Int, Section}, BitVector}
    set::Set{BitVector}
    sum::Dict{Tuple{Int, Section}, Int}
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

# function bases_to_dict( 
#         bases_array::BitArray{3};
#         sections::Vector{Section} = [Set(s) for s in 1:(size(bases_array)[3])]
#     )::Tuple{Dict{Tuple{Int,Section}, BitVector}, Set{BitVector}, Set{Section}}
#     #=
#     Helper for Bases constructor. Turns an (n x m x p) BitArray into a dictionary
#     (m, S(p)) => (n) where S(p) is the section for basis. Filters duplicate basis realizations
#     Also creates the set of realizations to facilitate future filtering.
#     =#

#     n_obs, n_knots, n_sections = size(bases_array)
#     reverse_bases_dict = Dict{BitVector, Tuple{Int,Section}}()

#     for section_id in 1:n_sections, knot in 1:n_knots
#         reverse_bases_dict[bases_array[:, knot, section_id]] = (knot, sections[section_id])
#     end

#     bases_dict = Dict(b=>basis for (basis, b) in reverse_bases_dict)
#     bases_set = Set(keys(reverse_bases_dict))
#     sections = Set(Section(j) for j in 1:n_sections)

#     return bases_dict, bases_set
# end

function BitMatrix(bases::Bases)::BitMatrix
    BitMatrix(hcat([basis for basis in values(bases.dict)]...))
end

function section_search(
        bases::Bases,
        Y::Vector{Float64};
        n_subsample::Int = 1000,
        m_subsample::Int = 1000,
    )::Section
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

    max_β = Y_subsample' * ones(n_subsample)/n_subsample # the current strength of the intercept
    section = Section()
    section_bases = BitMatrix(ones(n_subsample, m_subsample))
    
    while true
        β = abs.(
            sum(Y_subsample .* candidate_bases, dims=1) ./
            sum(candidate_bases, dims=1)
        )
        β[isnan.(β)] .= -Inf
        new_max_β, idx = findmax(β)
        _, knot, new_sectional_component = idx

        if new_max_β ≤ max_β # no improvement, return last section and bases
            # also catches the terminal case where a single one-way section is added twice in a row
            return section
        else
            push!(section, new_sectional_component)
            # section_bases = candidate_bases[:,:,new_sectional_component]
            section_basis = candidate_bases[:,:,new_sectional_component]
            candidate_bases .= section_bases .* one_way_bases
            max_β = new_max_β
        end
    end
end

function add_section!(
    bases::Bases, 
    section::Section, 
    section_bases::Union{BitMatrix,Nothing}=nothing
)::Bases
    # Adds the given sectional bases to the bases object, deduplicating anything that already exists

    if section == Section()
        return bases
    end

    if isnothing(section_bases)
        section_bases = prod(bases.one_way[:,:,[section...]], dims=3)[:,:,1]
    end

    n, m = size(section_bases)
    for (knot, basis) in enumerate(eachcol(section_bases))
        if basis ∉ bases.set
            b = (knot, section)
            push!(bases.set, basis)
            bases.dict[b] = basis
            bases.sum[b] = sum(basis)
            push!(bases.sections, section)
        end
    end
    return bases
end

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
    sections::Set{Section} = Set{Section}()
    # init_one_way=false
)::Bases
    one_way = build_one_way(X, knots)
    n,m,p = size(one_way)

    intercept_idx = (0, Section())
    intercept = BitVector(ones(n))

    # if init_one_way
    #     dict, set, sections = bases_to_dict(one_way)
    # else
    dict = Dict{Tuple{Int, Section}, BitVector}()
    set = Set{BitVector}()
    # end

    # add intercept
    push!(sections, Section())
    dict[intercept_idx] = intercept
    push!(set, intercept)
    set = Set([intercept])
    dict = Dict(intercept_idx => intercept)
    sums = Dict(intercept_idx => n)

    bases = Bases(knots, sections, one_way, dict, set, sums, n, m, p)
    for section in sections
        add_section!(bases, section)
    end

    return bases
end

function *(X::Bases, β::Dict{Tuple{Int, Section}, Float64})
    return sum(X[b]*β[b] for b in keys(β))
end

function add_basis!(bases::Bases, knot::Int, section::Section)::Bases
    idx = (knot, section)
    basis_vec = prod(bases.one_way[:,knot,[j for j in section]], dims=3)[:,1,1]
    push!(bases.set, basis_vec)
    bases.dict[idx] = basis_vec
    bases.sum[idx] = sum(basis_vec)

    push!(bases.sections, section)

    return bases
end

keys(bases::Bases) = (b for b in keys(bases.dict))

function getindex(bases::Bases, b::Tuple{Int, Section})::BitVector
    if b ∉ keys(bases.dict)
        add_basis!(bases, b...)
    end
    return bases.dict[b]
end



