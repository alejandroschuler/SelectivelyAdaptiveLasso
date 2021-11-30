#=
Data structures and methods to efficiently represent and work with realizations of HAL basis functions
=#

import Base: keys, getindex, setindex!, BitMatrix
import Combinatorics: powerset

Section = Set{Int}

mutable struct Bases
    knots::Matrix{Float64}
    one_way::BitArray{3}
    dict::Dict{Tuple{Int, Section}, BitVector}
    set::Set{BitVector}
    sum::Dict{Tuple{Int, Section}, Int}
    sections::Set{Section}
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

function bases_to_dict( 
        bases_array::BitArray{3};
        sections::Vector{Section} = [Set(s) for s in 1:(size(bases_array)[3])]
    )::Tuple{Dict{Tuple{Int,Section}, BitVector}, Set{BitVector}, Set{Section}}
    #=
    Helper for Bases constructor. Turns an (n x m x p) BitArray into a dictionary
    (m, S(p)) => (n) where S(p) is the section for basis. Filters duplicate basis realizations
    Also creates the set of realizations to facilitate future filtering.
    =#

    n_obs, n_knots, n_sections = size(bases_array)
    reverse_bases_dict = Dict{BitVector, Tuple{Int,Section}}()

    for section_id in 1:n_sections, knot in 1:n_knots
        reverse_bases_dict[bases_array[:, knot, section_id]] = (knot, sections[section_id])
    end

    bases_dict = Dict(b=>basis for (basis, b) in reverse_bases_dict)
    bases_set = Set(keys(reverse_bases_dict))
    sections = Set(Section(j) for j in 1:n_sections)

    return bases_dict, bases_set
end

function Bases(X::Matrix{Float64}, knots::Matrix{Float64} = X; init_one_way=false)::Bases
    n,p = size(X)
    one_way = build_one_way(X, knots)

    intercept_idx = (0, Section())
    intercept = BitVector(ones(n))

    if init_one_way
        dict, set, sections = bases_to_dict(one_way)
    else
        dict = Dict{Tuple{Int, Section}, BitVector}()
        set = Set{BitVector}()
        sections = Set{Section}()
    end

    # add intercept
    dict[intercept_idx] = intercept
    push!(set, intercept)
    push!(sections, Section())

    sums = Dict(b=>sum(v) for (b,v) in dict)

    Bases(knots, one_way, dict, set, sums, sections)
end

getindex(bases::Bases, b::Tuple{Int, Section})::BitVector = bases.dict[b]
keys(bases::Bases) = (b for b in keys(bases.dict))

function BitMatrix(bases::Bases)::BitMatrix
    BitMatrix(hcat([basis for basis in values(bases.dict)]...))
end

function section_search(
        bases::Bases,
        Y::Vector{Float64};
    )::Tuple{BitMatrix, Section}
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
    each candidate section. This is slightly slower because of the sum and doesn't appear to increase 
    "performance", but perhaps worth revisting later. I also tried a recursive implementation but looping 
    is faster.
    =# 
    
    section = Section()
    max_β = 0.0
    bases_array = copy(bases.one_way)
    
    while true
        new_max_β, idx = findmax(  
            abs.(
                sum(Y .* bases_array, dims=1) ./
                sum(bases_array, dims=1)
            )
        )
        new_sectional_component = idx[3]

        if new_max_β ≤ max_β # we have a candidate section
            # also catches the terminal case where a single one-way section is added twice in a row
            return bases_array[:,:,new_sectional_component], section
        else
            bases_array .= bases_array[:,:,new_sectional_component] .* bases.one_way
            max_β = new_max_β
            push!(section, new_sectional_component)
        end
    end
end

function add_section!(bases::Bases, section_bases::BitMatrix, section::Section)::Bases
    # Adds the given sectional bases to the bases object
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

function expand!(bases::Bases, Y::Vector{Float64})::Bases
    #=
    Add a section to the set of bases. Uses a heuristic search to find the section most useful to predict Y. If said 
    section is already included, pick the first section (in increasing order of interactions) that isn't. This is 
    unlikely to happen since sections that have already been picked are residualized out.
    =#
    bases_array, section = section_search(bases, Y)
    print(section, bases.sections)
    if section in bases.sections # already included, pick something random
        n,m,p = size(bases.one_way)
        print("picking random section\n")
        for section in (Section(s) for s in powerset(1:p))
            if section ∉ bases.sections
                section_idx = [sj for sj in section]
                bases_array = prod(bases.one_way[:,:,section_idx], dims=3)[:,:,1]
                return add_section!(bases, bases_array, section)
            end
        end
    end
    return add_section!(bases, bases_array, section)
end

function instantiate(bases::Bases, X::Matrix{Float64})::Bases
    instantiated = Bases(X, bases.knots)
    for section in bases.set
    end
    ## add additional sections present in bases
end