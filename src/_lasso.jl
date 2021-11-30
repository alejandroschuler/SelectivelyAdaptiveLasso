function lasso_ccd_update(β, x, residual, λ)
    ρ = (residual' * x)::Float64
    s = sum(x.^2)
    return (soft_thresh(β*s + ρ, λ) / s)::Float64
end

function lasso_coordinate_descent(
        X::AbstractMatrix,
        Y::Vector{Float64};
        λ::Real = 1,
        β::Vector{Float64} = zeros(size(X)[2]),
        tol::Number = 1e-6,
    )
    
    n, p = size(X)
    λ *= n # rescale λ so that Yᵢ∈[-1,1], λ=1 => β=0
    
    residual = Y - X*β # can pass from prev iter
    loss(residual, β) = (1/n) * ( (1/2)sum(residual.^2) + λ*sum(abs.(β)) )
    
    loss_0 = loss(residual, β)
    cycle_loss = [loss_0]
        
    cycles = 0
    while cycles < 100 
        # print("$(keys(β))\n")
        cycles += 1
        
        for b in 1:p 
            β̃b = lasso_ccd_update(β[b], X[:,b], residual, λ)
            if β̃b - β[b] ≠ 0
                residual -= X[:,b] * (β̃b - β[b])
                β[b] = β̃b
                push!(iter_loss, loss(residual, β))
            end
        end
        
        push!(cycle_loss, loss(residual, β))
        if ( abs(cycle_loss[end] - cycle_loss[end-1]) < tol ) # check loss convergence
            break
        end
        
    end
    
    return β, cycle_loss
end