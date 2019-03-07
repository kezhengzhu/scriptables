function us = simpmie(rs,r,a)

cmie = r/(r-a) * r/a ^ (a/(r-a));
us = cmie .* (rs.^(-r) - rs.^(-a));

end