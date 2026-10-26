function [state, record] = link_outcome(state, ctx, record, t)
    %LINK_OUTCOME Transport-block success against the MCS planned for uplink slot t
    %
    % The block decodes when the true rate covers the planned rate, and
    % outer-loop link adaptation moves the rate backoff by the asymmetric
    % steps that settle at the target block error rate
    %
    % Args:
    %   state: Method state with plan and backoff
    %   ctx: Run context from nr_run
    %   record: Slot record with the true rates
    %   t: Uplink slot index
    %
    % Returns:
    %   state: State with the updated backoff
    %   record: Record with planned rates, throughput, and NACK count

    planned = state.plan(:, t);
    record.planned = planned;

    if any(isnan(planned))
        record.throughput = nan(size(planned));
        return;
    end

    success = record.rates >= planned - 1e-12;
    record.throughput = planned .* success;
    record.nack = sum(~success & planned > 0);
    bler = ctx.cfg.link.targetBler;
    step = ctx.cfg.link.ollaStep;
    state.backoff = max(state.backoff - step * success + ...
        step * (1 - bler) / bler * (~success & planned > 0), 0);
end
