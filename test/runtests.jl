import Random
import Statistics: mean
import SelectivelyAdaptiveLasso as SAL
using Test

@testset "FeatureVector" begin
    n = 100
    U = rand(n)
    X = SAL.FeatureVector(U)
    I = Random.shuffle(1:n)[1:10]

    @test all(U[I] == X.sorted[X.raw_to_sorted[I]])
    @test all(U[X.sorted_to_raw[I]] == X.sorted[I])
    @test X.sorted[sort(X.raw_to_sorted[I])] == U[I[SAL.sortperm_subset(X,I)]]
    @test issorted(X.sorted[X.raw_to_sorted[I[SAL.sortperm_subset(X,I)]]], rev=true)


end

@testset "basis selection" begin
for i in 1:100
    n=100
    c = 0.4
    I = collect(1:2:n) # must be increasing to keep X sorted
    X = sort(rand(n), rev=true)
    Y = Vector{Float64}(X.>c)

    Ŷ(X,Y) = X*mean(Y[X])
    sses_brute(X,Y) = [sum((Y.-Ŷ(X.≥xi,Y)).^2) for xi in X]
    sses_clever = SAL.sse_indicator_ols(Y)

    @test all(sses_brute(X,Y) .≈ sses_clever)
    @assert issorted(X[I], rev=true)

    Ĩ, Ỹ, _, x̃, l̃ = SAL.best_split_sorted(I,X[I],Y[I])
    @test Set(Ĩ) == Set((1:n)[X.≥c]) ∩ Set(I)
    @test all(Ỹ .== 1)
    @test x̃ == min(X[I][X[I].≥c]...)
    @test l̃ ≈ min(sses_brute(X[I], Y[I])...)


    n,p = 100,3
    j° = p
    U = rand(n,p)
    I = collect(1:2:n) # must be increasing to keep X sorted
    X = [SAL.FeatureVector(x) for x in eachcol(U)]
    Y = Vector{Float64}(U[:,j°] .≥ c)

    Ĩ, Ỹ, _, (j̃, x̃), l̃ = SAL.interact_basis(I, X, Y[I])
    @test Set(Ĩ) == Set((1:n)[U[:,j°].≥c]) ∩ Set(I)
    @test all(Ỹ .== 1)
    @test x̃ == min(U[I,j°][U[I,j°].≥c]...)
    @test j̃ == j°
    @test l̃ ≈ min(sses_brute(U[I,j°], Y[I])...)


    j°° = 1
    Y = Vector{Float64}((U[:,j°] .≥ c) .& (U[:,j°°] .≥ c))

    # index, basis = SAL.basis_search(X, Y, feat_n=p)
    # js, xs = zip([(j,x) for (j,x) in index]...)

    # basis_ints = Set(Vector{Int}(basis.nzind))
    # Y_ints = Set((1:n)[Y.==1])
    # basis_sans_Y = setdiff(basis_ints, Y_ints)
    # Y_sans_basis = setdiff(Y_ints, basis_ints)
    # if length(index) > 2
    #     println("derp")
    # end
    # if !isempty(basis_sans_Y) | !isempty(Y_sans_basis)
    #     println((basis_sans_Y, Y_sans_basis))
    #     println(U[[Y_sans_basis...], :])
    #     println(index)
    # end
    # @test basis_ints == Y_ints
    # # @test Set(js) == Set([j°, j°°])


    index, basis = SAL.basis_search_random(X)
    n_val = 20
    U_val = rand(n_val,p)
    X_val = [SAL.FeatureVector(x) for x in eachcol(U_val)]
    basis_val = SAL.build_basis(X_val, index)

    bool_basis = reduce(.*, [U[:,j] .≥ c for (j,c) in index])
    bool_basis_val = reduce(.*, [U_val[:,j] .≥ c for (j,c) in index])
    @test Set(Vector{Int}(basis.nzind)) == Set((1:n)[bool_basis])
    @test Set(Vector{Int}(basis_val.nzind)) == Set((1:n_val)[bool_basis_val])
end
end