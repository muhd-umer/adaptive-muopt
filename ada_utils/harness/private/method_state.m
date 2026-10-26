function state = method_state(name, ctx, active)
    %METHOD_STATE State every method keeps: MCS plan, link adaptation, and control link
    %
    % Args:
    %   name: Method name
    %   ctx: Run context from nr_run
    %   active: Logical row of scheduled UEs
    %
    % Returns:
    %   state: Common state fields

    U = ctx.env.U;
    state = struct('name', name, 'active', active, 'evaluations', 0);
    state.plan = nan(U, numel(ctx.slots) + 2);
    state.backoff = zeros(U, 1);
    state.ctrl = ctrl_init(U, ctx.cfg.control);
    state.pendingUntil = -inf(1, U);
end
