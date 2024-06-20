function [sigma] = AR(p, rho)
sigma = zeros(p);
for i = 1:p
    for j = 1:p
        sigma(i, j) = rho^(abs(i - j));
    end
end
end