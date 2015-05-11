function [class] = convertToClass(value)
threshold = 0.000;

if value >= threshold
    class = 1;
% elseif value <= -threshold
%     class = -1;
else
    class = -1;
end

end

