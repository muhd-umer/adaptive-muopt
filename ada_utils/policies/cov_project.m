function [R, factors] = cov_project(R, budgets, floorValue)
    %COV_PROJECT UE-side projection of decoded covariances onto the floor and budget
    %
    % Each UE pools the eigenvalues of its per-PRG covariances and projects
    % them onto {x >= floor, sum(x) <= budget}, the rule the gNB also applies
    % when it predicts the deployed design, and the transmit factor is the
    % eigen-decomposition sorted by decreasing eigenvalue
    %
    % Args:
    %   R: U-by-G cell of covariances
    %   budgets: Per-UE energy budget summed over the G blocks, zero to silence
    %   floorValue: Per-UE eigenvalue floor
    %
    % Returns:
    %   R: Projected covariances
    %   factors: Matching transmit factors V * sqrt(diag(values))

    [U, G] = size(R);
    floorValue = floorValue(:) .* ones(U, 1);
    factors = cell(U, G);

    for u = 1:U
        bases = cell(1, G);
        values = cell(1, G);

        for g = 1:G
            [V, D] = eig(0.5 * (R{u, g} + R{u, g}'));
            bases{g} = V;
            values{g} = real(diag(D));
        end

        pooled = vertcat(values{:});

        if budgets(u) <= 0
            projected = zeros(size(pooled));
        else
            projected = waterfill(pooled, budgets(u), min(floorValue(u), ...
                budgets(u) / numel(pooled)));
        end

        offset = 0;

        for g = 1:G
            count = numel(values{g});
            part = projected(offset + (1:count));
            offset = offset + count;
            [part, order] = sort(part, 'descend');
            V = bases{g}(:, order);
            R{u, g} = 0.5 * (V * diag(part) * V' + (V * diag(part) * V')');
            factors{u, g} = V * diag(sqrt(part));
        end

    end

end

function projected = waterfill(pooled, total, floorValue)
    shifted = pooled - floorValue;
    budget = total - floorValue * numel(pooled);
    positive = max(shifted, 0);

    if sum(positive) <= budget
        projected = floorValue + positive;
        return;
    end

    sorted = sort(shifted, 'descend');
    cumulative = cumsum(sorted);
    indices = (1:numel(sorted))';
    active = find(sorted - (cumulative - budget) ./ indices > 0, 1, 'last');
    threshold = (cumulative(active) - budget) / active;
    projected = floorValue + max(shifted - threshold, 0);
end
