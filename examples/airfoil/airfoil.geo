
h = 0.1;
r = 6;
p = 0;
asym = 0.5;

sf = 0.005;
sc = 0.1;


Point(1) = {0.5, 0, 0, sf};

Point(2) = {-0.2, h*(1 + asym), 0, sf};

Point(3) = {-0.5 + p, h/2*(1 + asym), 0, sf};

Point(4) = {-0.5, 0, 0, sf};

Point(5) = {-0.5 + p, -h/2*(1 - asym), 0, sf};

Point(6) = {-0.2, -h*(1 - asym), 0, sf};

BSpline(1) = {1, 2, 3, 4};
BSpline(2) = {4, 5, 6, 1};

Point(7) = {r, 0, 0, sc};
Point(8) = {0, r, 0, sc};
Point(9) = {-r, 0, 0, sc};
Point(10) = {0, -r, 0, sc};
Point(11) = {0, 0, 0, sf};

Circle(3) = {7, 11, 8};
Circle(4) = {8, 11, 9};
Circle(5) = {9, 11, 10};
Circle(6) = {10, 11, 7};

Curve Loop(1) = {1, 2};
Curve Loop(2) = {3, 4, 5, 6};

Plane Surface(1) = {1, 2};



//+
Extrude {0, 0, 0.5} {
  Surface{1}; Layers {1}; Recombine;
}

//+
Physical Surface("sides", 39) = {38, 1};
//+
Physical Surface("farfield", 40) = {33, 29, 25, 37};
//+
Physical Surface("wing", 41) = {17, 21};
//+
Physical Volume("internal", 42) = {1};

Mesh 3;

Save "airfoil.su2";
