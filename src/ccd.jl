#=
Cyclical Coordinate Descent for lasso with binary features
=#

function soft_thresh(
        z::Real,
        λ::Real
    )::Float64
    if abs(z) ≤ λ
        return 0
    elseif z > 0
        return z - λ
    else
        return z + λ
    end
end

function coordinate_descent(
        X::Bases,
        Y::Vector{Float64};
        λ::Real = 1,
        β::Dict{BasisIndex, Float64} = Dict{BasisIndex, Float64}(INTERCEPT => 0.0),
        tol::Number = 1e-4,
    )::Tuple{Dict{BasisIndex, Float64}, Vector{Float64}, Vector{Float64}}
   
    #=
    A fast coordinate descent algorithm for lasso with binary features. Can be warm-started by passing β. 
    =#
    
    n = length(Y)

    residual = Y - X*β # can pass from prev iter 
    loss(residual, β) = (1/n) * ( (1/2)sum(residual.^2) + λ*sum(abs.(values(β))) )
    
    loss_0 = loss(residual, β)
    cycle_loss = [loss_0]
    
    loop_bases = keys(X)
    
    @inbounds while true
        if length(cycle_loss) > 10000
            print(cycle_loss[(end-10):end])
            break
        end
        
        # update cycle
        for b in loop_bases
            βb = get(β, b, 0.0)
            sb = X.sum[b]
            ρ = residual * X[b]

            β̃b = soft_thresh(βb*sb + ρ, λ) / sb
            Δβb = β̃b - βb
            
            if Δβb ≠ 0 # tried with deleting β[b] if βb=0, faster as-is
                β[b] = β̃b
                for i in X[b].nzind
                    residual[i] -= Δβb
                end
            end
        end
        
        # check convergence and set loop over active set
        push!(cycle_loss, loss(residual, β))
        if ( abs(cycle_loss[end] - cycle_loss[end-1]) < tol ) # check loss convergence
            if loop_bases == keys(X)
                break
            end
            loop_bases = keys(X)
        else
            loop_bases = keys(β)
        end
        
    end
    
    β = Dict(
        b=>βb for (b,βb) in β 
        if (βb ≠ 0.0) | ( b == INTERCEPT )
    )
    return β, residual, cycle_loss
end