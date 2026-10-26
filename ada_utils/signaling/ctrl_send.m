function ctrl = ctrl_send(ctrl, slot, ue, kind, bits, payload)
    %CTRL_SEND Queue a control message that activates after the link delay
    %
    % Args:
    %   ctrl: Control link from ctrl_init
    %   slot: Slot in which the scheduler decided
    %   ue: Destination UE
    %   kind: Message kind, e.g. 'dci01' or 'covariance'
    %   bits: Exact payload bits charged to the message
    %   payload: Encoded content the UE will decode
    %
    % Returns:
    %   ctrl: Updated control link

    link = ctrl.link;
    due = slot + link.delay;
    late = rand() < get_opt(link, 'pLate', 0);
    dropped = rand() < get_opt(link, 'pDrop', 0);

    if late
        due = due + get_opt(link, 'lateSlots', 1);
        ctrl.late(ue) = ctrl.late(ue) + 1;
    end

    ctrl.queue(end + 1) = struct('ue', ue, 'sent', slot, 'due', due, ...
        'kind', kind, 'bits', bits, 'payload', payload, 'dropped', dropped);
    ctrl.messages(ue) = ctrl.messages(ue) + 1;
    ctrl.bits(ue) = ctrl.bits(ue) + bits;
    ctrl.dropped(ue) = ctrl.dropped(ue) + dropped;
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
