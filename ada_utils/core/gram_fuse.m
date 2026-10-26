function K = gram_fuse(K, J, A, beta)
    %GRAM_FUSE Fold pilot-learned information into a Gram-equivalent belief
    %
    % Information learned at a known transmit factor A satisfies
    % J - I = A' * Gram * A. That pins the Gram matrix only inside range(A),
    % so the belief is corrected there by the minimum-norm update
    % K + pinv(A)' * beta * (J - I - A' * K * A) * pinv(A) and left untouched
    % in the unexcited directions. A full-rank A with beta = 1 replaces the
    % belief outright, a rank-deficient A such as precoded DMRS refreshes only
    % the excited subspace, and beta < 1 is exponential forgetting. The result
    % is the Gram-equivalent statistic the optimizer needs; the channel matrix
    % itself is never reconstructed
    %
    % Args:
    %   K: Ltot-by-Ltot prior belief, or empty for no prior
    %   J: r-by-r information matrix learned at A
    %   A: Ltot-by-r known transmit factor of the pilots
    %   beta: Weight of the new observation in (0, 1]
    %
    % Returns:
    %   K: Updated Hermitian positive-semidefinite belief

    Ltot = size(A, 1);

    if isempty(K)
        K = zeros(Ltot);
    end

    excited = vecnorm(A, 2, 1) > 0;
    A = A(:, excited);
    J = J(excited, excited);

    if isempty(A)
        return;
    end

    pseudo = pinv(A);
    innovation = (J - eye(size(J))) - A' * K * A;
    K = K + pseudo' * (beta * 0.5 * (innovation + innovation')) * pseudo;
    K = 0.5 * (K + K');
    [V, D] = eig(K);
    K = V * diag(max(real(diag(D)), 0)) * V';
    K = 0.5 * (K + K');
end
