
SetFactory("OpenCASCADE");
Circle(1) = {0, -0, 0, 2.0, 0, 2*Pi};
Circle(2) = {0, -0, 0, 0.1, 0, 2*Pi};


Curve Loop(1) = {1};
Curve Loop(2) = {2};

Plane Surface(1) = {1, 2};


Extrude {0, 0, 1.0} {
    Surface{1}; Layers{40}; Recombine;
}


Field[1] = MathEval;
Field[1].F = "max(0.08*sqrt(x^2 + y^2), 0.02)";

Background Field = 1;

//+
Physical Surface("sides", 7) = {1, 4};
//+
Physical Surface("circle", 8) = {3};
//+
Physical Surface("farfield", 9) = {2};
//+
Physical Volume("internal", 10) = {1};

//Recombine Surface {1};

Mesh 3;

Save "mesh.su2";

