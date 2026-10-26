function index = next_occasion(ctx, slot)
    %NEXT_OCCASION Index of the first uplink slot at or after a slot
    %
    % Args:
    %   ctx: Run context from nr_run
    %   slot: Slot number
    %
    % Returns:
    %   index: Uplink slot index, one past the horizon when none is left

    index = find(ctx.slots >= slot, 1);

    if isempty(index)
        index = numel(ctx.slots) + 1;
    end

end
