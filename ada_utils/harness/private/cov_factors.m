function factors = cov_factors(R)
    %COV_FACTORS Transmit factors V * sqrt(D) of covariances, strongest direction first
    %
    % Args:
    %   R: Cell array of Hermitian positive-semidefinite covariances
    %
    % Returns:
    %   factors: Cell array of factors F with R = F * F'

    factors = cell(size(R));

    for i = 1:numel(R)
        [V, D] = eig(0.5 * (R{i} + R{i}'));
        [values, order] = sort(max(real(diag(D)), 0), 'descend');
        factors{i} = V(:, order) * diag(sqrt(values));
    end

end
