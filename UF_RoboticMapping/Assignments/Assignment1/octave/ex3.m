
% To test the functions implemented in question 3
% All the outputs should be [0 0 0]
% The last value can be close to 0 or -6.28

%%%%%% QUESTION 3(A) %%%%%%
fprintf('\nQUESTION 3A: Implement v2t and t2v\n');

% Special values
a = [1 2 0]; 
disp(t2v(v2t(a)) - a);

a = [1 2 pi/2];
disp(t2v(v2t(a)) - a);

a = [1 2 pi];
disp(t2v(v2t(a)) - a);

a = [1 2 1.5*pi];
disp(t2v(v2t(a)) - a);

a = [1 2 2*pi];
disp(t2v(v2t(a)) - a);

% First quad
a = [1 2 0.5];
disp(t2v(v2t(a)) - a);

% Second quad
a = [1 2 1.2];
disp(t2v(v2t(a)) - a);

% Third quad
a = [1 2 2.0];
disp(t2v(v2t(a)) - a);

% Fourth quad
a = [1 2 2.6];
disp(t2v(v2t(a)) - a);


%%%%%% QUESTION 3(B) %%%%%%
fprintf('\nQUESTION 3B: Given pose p1 and p2, find the relative transformation from p1 to p2\n');

% Convert both poses to homogeneous transformations
% p1 -> M1
% p2 -> M2
% Find the transformation matrix T such that M2 = t * M1
% Therefore, t = M2 * M1'
p1 = [1 2 0.5];
p2 = [3 4 -1.3];

M1 = v2t(p1);
M2 = v2t(p2);

t = M2 * inv(M1);

M1_transformed = t2v(t * M1);
disp(M1_transformed - p2);


%%%%%% QUESTION 3(C) %%%%%%
fprintf('\nQUESTION 3C: Given a robot pose p1 and an <x,y> observation z of a landmark relative to p1. Compute the location of the landmark.\n');

% (NOT SURE)
% Convert both into the homogeneous coordinate system
p1 = [1 1 pi/2];
p1_homo = p1 / p1(3);

z = [2 0];
z_homo = [z 1];

loc = p1_homo + z_homo;
disp(loc);

