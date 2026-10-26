function [R, refit] = fit_shared(R, etotal, active)
    %FIT_SHARED Scale every scheduled UE back onto a shared budget it exceeds
    %
    % Args:
    %   R: U-by-G cell of covariances
    %   etotal: Shared per-RE energy budget
    %   active: Logical row of scheduled UEs
    %
    % Returns:
    %   R: Covariances within the budget
    %   refit: Whether a rescale was needed

    G = size(R, 2);
    energy = sum(cellfun(@(B) real(trace(B)), R(:))) / G;
    refit = energy > etotal * (1 + 1e-9);

    if refit
        R(active, :) = cellfun(@(B) B * (etotal / energy), R(active, :), 'UniformOutput', false);
    end

end
