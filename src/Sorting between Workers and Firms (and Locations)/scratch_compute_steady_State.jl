# Define δ_jxy = δ when jxy is a valid job, 1 otherwise
δ_jxy = ones(prim.n_j, prim.n_x, prim.n_y)
for j ∈ 1:prim.n_j
    for x ∈ 1:prim.n_x
        for y ∈ 1:prim.n_y
            if res.S_move[j,j,x,y] > 0
                δ_jxy[j, x, y] = prim.δ
            end
        end
    end
end

# Compute for each 