function [ctrl, delivered, lost] = ctrl_deliver(ctrl, slot)
    %CTRL_DELIVER Pop the control messages whose activation slot has arrived
    %
    % Messages activate in arrival order, so a newer decision that lands in
    % the same slot overrides an older one for the same UE
    %
    % Args:
    %   ctrl: Control link from ctrl_init
    %   slot: Current slot
    %
    % Returns:
    %   ctrl: Updated control link
    %   delivered: Messages the UEs apply in this slot
    %   lost: Messages that were due now but never reached the UE

    if isempty(ctrl.queue)
        delivered = ctrl.queue;
        lost = ctrl.queue;
        return;
    end

    due = [ctrl.queue.due] <= slot;
    arrived = ctrl.queue(due);
    ctrl.queue = ctrl.queue(~due);
    [~, sequence] = sort([arrived.sent]);
    arrived = arrived(sequence);
    lost = arrived([arrived.dropped]);
    delivered = arrived(~[arrived.dropped]);

    for i = 1:numel(delivered)
        ue = delivered(i).ue;
        ctrl.delivered(ue) = ctrl.delivered(ue) + 1;
    end

end
