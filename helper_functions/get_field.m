function value = get_field(s, field_name, default)
    % Helper function to get field from struct
    if isfield(s, field_name)
        value = s.(field_name);
    else
        value = default;
    end
end