function [v] = t2v(t)

% Takes a transformation and returns the pose

v(1) = t(1, 3);
v(2) = t(2, 3);

cos_theta = t(1, 1);
sin_theta = t(2, 1);

if (cos_theta > 0 && sin_theta > 0)
	v(3) = acos(cos_theta);
elseif (cos_theta < 0 && sin_theta > 0)
	v(3) = acos(cos_theta);
elseif (cos_theta > 0 && sin_theta < 0)
	v(3) = asin(sin_theta);
else
	v(3) = asin(sin_theta);
endif

end
