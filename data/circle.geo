

SetFactory("OpenCASCADE");
Circle(1) = {0, -0, 0, 0.5, 0, 2*Pi};

Curve Loop(1) = {1};

Plane Surface(1) = {1};


Extrude {0, 0, 1} {
  Surface{1};
}
Physical Surface("bottom", 4) = {1};

Physical Surface("top", 5) = {3};

Physical Surface("side", 6) = {2};

Physical Volume("internal", 7) = {1};

Field[1] = MathEval;
Field[1].F = "0.02";

Background Field = 1;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh 3;

Save "mesh.su2";
