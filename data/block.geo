//+
Point(1) = {-0, -0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {4, 1, 2, 3} = 2 Using Progression 1;
//+
Transfinite Surface {1};
//+
Recombine Surface {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {1000}; Recombine;
}

Physical Surface("side", 27) = {13, 21, 25, 17};
//+
Physical Surface("bottom", 28) = {1};
//+
Physical Surface("top", 29) = {26};

Physical Volume("internal", 30) = {1};

Mesh 3;

Save "mesh.su2";
