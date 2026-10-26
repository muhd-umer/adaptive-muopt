function value = page_logdet(M)
    %PAGE_LOGDET Log-determinant of every Hermitian positive-definite page
    %
    % Args:
    %   M: n-by-n-by-T stack of Hermitian positive-definite matrices
    %
    % Returns:
    %   value: 1-by-T natural log-determinants

    if isempty(M)
        value = zeros(1, size(M, 3));
        return;
    end

    singular = pagesvd(M);
    value = reshape(sum(log(max(singular, realmin)), 1), 1, []);
end
