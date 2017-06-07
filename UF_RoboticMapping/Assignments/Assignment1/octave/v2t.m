function [t] = v2t(pose)

% Takes a robot pose and returns the homogeneous transformation

x = pose(1);
y = pose(2);
theta = pose(3);

t = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];

end
