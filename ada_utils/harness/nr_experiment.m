function [results, env] = nr_experiment(cfg, spec)
    %NR_EXPERIMENT Run every realization of an NR experiment, in parallel when possible
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %   spec: Objective passed to nr_run
    %
    % Returns:
    %   results: Cell array of nr_run outputs, one per realization
    %   env: Environment shared by every realization

    env = nr_setup(cfg);
    R = cfg.realizations;
    results = cell(1, R);
    usePool = cfg.parallel && R > 1 && ~isempty(ver('parallel')) && ...
        license('test', 'Distrib_Computing_Toolbox');

    if usePool

        if isempty(gcp('nocreate'))
            parpool('Processes');
        end

        parfor r = 1:R
            results{r} = nr_run(cfg, env, spec, r);
        end

    else

        for r = 1:R
            results{r} = nr_run(cfg, env, spec, r);
        end

    end

end
