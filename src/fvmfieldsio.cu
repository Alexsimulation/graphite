#include "fvmfieldsio.cuh"



int FieldsWriteVtu(
	const char* filename,
	cuMesh* mesh_in,
	const FvmField** vars,
	const size_t nvars
) {
	// Write vtu file
	cuMeshBase* mesh = mesh_in->_cpu;


	FILE* file = fopen(filename, "wt");


	// Write mesh
	fprintf(file, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
	fprintf(file, "  <UnstructuredGrid>\n");
	fprintf(file, "    <Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\">\n", mesh->n_nodes, mesh->n_cells);

	// Write the cells data
	fprintf(file, "      <CellData>\n");

	for (int i=0; i<nvars; ++i) {
		const FvmField* var = vars[i];

		switch (FvmFieldGetType(var)) {
		case SCALAR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				scalar v = ((scalar*)var->_cpu->value)[i];
				if (isnan(v)) {
					fprintf(file, "          nan\n");
				} else {
					fprintf(file, "          %lf\n", v);
				}
			}
			break;
		case VECTOR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"3\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				fprintf(file, "         ");

				vector v = ((vector*)var->_cpu->value)[i];
				if (isnan(v.x)) {
					fprintf(file, " nan");
				} else {
					fprintf(file, " %lf", v.x);
				}
				if (isnan(v.y)) {
					fprintf(file, " nan");
				} else {
					fprintf(file, " %lf", v.y);
				}
				if (isnan(v.z)) {
					fprintf(file, " nan\n");
				} else {
					fprintf(file, " %lf\n", v.z);
				}
			}
			break;
		}

		fprintf(file, "        </DataArray>\n");

		#ifdef FVM_OUTPUT_GRAD_LIM

		switch (var->_cpu->type) {
		case SCALAR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s_grad\" NumberOfComponents=\"3\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				vector v_g = ((vector*)var->_cpu->gradient)[i];
				scalar v_gval[3] = {v_g.x, v_g.y, v_g.z};
				fprintf(file, "         ");
				for (int j=0; j<3; ++j) {
					if (isnan(v_gval[j])) {
						fprintf(file, " nan");
					} else {
						fprintf(file, " %lf", v_gval[j]);
					}
				}
				fprintf(file, "\n");
			}
			break;
		case VECTOR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s_grad\" NumberOfComponents=\"9\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				tensor v_g = ((tensor*)var->_cpu->gradient)[i];

				scalar v_gval[9] = {
					v_g.u.x, v_g.v.x, v_g.w.x,
					v_g.u.y, v_g.v.y, v_g.w.y,
					v_g.u.z, v_g.v.z, v_g.w.z
				};
				fprintf(file, "         ");
				for (int j=0; j<9; ++j) {
					if (isnan(v_gval[j])) {
						fprintf(file, " nan");
					} else {
						fprintf(file, " %lf", v_gval[j]);
					}
					if ((j != 8)&(j != 0)&((j+1) % 3 == 0)) {
						fprintf(file, "\n         ");
					}
				}
				fprintf(file, "\n");
			}
			break;
		}

		fprintf(file, "        </DataArray>\n");

		switch (var->_cpu->type) {
		case SCALAR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s_lim\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				scalar v_lim = ((scalar*)var->_cpu->limiter)[i];
				if (isnan(v_lim)) {
					fprintf(file, "         nan\n");
				} else {
					fprintf(file, "         %lf\n", v_lim);
				}
			}
			break;
		case VECTOR:
			fprintf(file, "        <DataArray type=\"Float32\" Name=\"%s_lim\" NumberOfComponents=\"3\">\n", var->name);
			for (size_t i = 0; i < var->_cpu->size; ++i) {
				vector v_lim = ((vector*)var->_cpu->limiter)[i];

				scalar v_clim[3] = {
					v_lim.x, v_lim.y, v_lim.z
				};
				fprintf(file, "         ");
				for (int j=0; j<3; ++j) {
					if (isnan(v_clim[j])) {
						fprintf(file, " nan");
					} else {
						fprintf(file, " %lf", v_clim[j]);
					}
				}
				fprintf(file, "\n");
			}
			break;
		}

		fprintf(file, "        </DataArray>\n");

		#endif
	}

	fprintf(file, "      </CellData>\n");

	// Write nodes
	fprintf(file, "      <Points>\n");
	fprintf(file, "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");

	for (size_t i = 0; i< mesh->n_nodes; ++i)
		fprintf(file, "          %lf %lf %lf\n", mesh->nodes[i].x, mesh->nodes[i].y, mesh->nodes[i].z);

	fprintf(file, "        </DataArray>\n");
	fprintf(file, "      </Points>\n");

	// Write cells
	fprintf(file, "      <Cells>\n");

	// Write cell nodal connectivity, compute cell offsets
	fprintf(file, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
	uint* cell_offsets = cpuAlloc(uint, mesh->n_cells);
	uint offset = 0;
	for (size_t e = 0; e < mesh->n_cells; ++e) {
		// Get the unique nodes of this elements
		uint nodes[64];	// max node size is 64
		uint nodes_size = 0;
		nodes[0] = 999999999;

		// number_of faces face_number_of
		size_t elem_size = cuMeshGetCellSize(mesh, e);

		uint* elem_start = cuMeshGetCellStart(mesh, e);
		uint* elem_end = cuMeshGetCellEnd(mesh, e);
		for (uint* f = elem_start; f != elem_end; ++f) {
			uint face = *f;

			uint face_size = cuMeshGetFaceSize(mesh, face);
			uint* face_start = cuMeshGetFaceStart(mesh, face);
			uint* face_end = cuMeshGetFaceEnd(mesh, face);
			for (uint* n = face_start; n != face_end; ++n) {
				uint node = *n;
				// Check if node in nodes[], add it if true
				for (size_t ni = 0; ni < 64; ++ni) {
					if (node == nodes[ni]) break;
					if (ni == nodes_size) {
						nodes[ni] = node;
						nodes_size++;
						break;
					}
					if (node < nodes[ni]) {
						memmove(nodes + ni + 1, nodes + ni, sizeof(uint) * (nodes_size - ni));
						nodes[ni] = node;
						nodes_size++;
						break;
					}
				}
			}
		}
		// Write the nodes

		fprintf(file, "          ");
		for (size_t n = 0; n < nodes_size; ++n)
			fprintf(file, " %u", nodes[n]);

		fprintf(file, "\n");

		offset += nodes_size;
		cell_offsets[e] = offset;
	}
	fprintf(file, "        </DataArray>\n");

	// Write the cell offsets
	fprintf(file, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
	for (size_t e = 0; e < mesh->n_cells; ++e) {
		fprintf(file, "          %u\n", cell_offsets[e]);
	}
	free(cell_offsets);
	fprintf(file, "        </DataArray>\n");

	// Write the cell types
	fprintf(file, "        <DataArray type=\"Int64\" Name=\"types\" format=\"ascii\">\n");
	for (size_t e = 0; e < mesh->n_cells; ++e) {
		fprintf(file, "          42\n");
	}
	fprintf(file, "        </DataArray>\n");

	// Write cells faces and compute face offsets
	fprintf(file, "        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n");

	uint* face_offsets = cpuAlloc(uint, mesh->n_cells);
	offset = 0;
	for (size_t e = 0; e < mesh->n_cells; ++e) {
		// number_of faces face_number_of
		size_t elem_size = cuMeshGetCellSize(mesh, e);
		fprintf(file, "          %zu", elem_size);
		offset++;

		uint* elem_start = cuMeshGetCellStart(mesh, e);
		uint* elem_end = cuMeshGetCellEnd(mesh, e);
		for (uint* f = elem_start; f != elem_end; ++f) {
			uint face = *f;

			uint face_size = cuMeshGetFaceSize(mesh, face);
			uint* face_start = cuMeshGetFaceStart(mesh, face);
			uint* face_end = cuMeshGetFaceEnd(mesh, face);
			fprintf(file, " %u", face_size);
			offset++;
			for (uint* n = face_start; n != face_end; ++n) {
				fprintf(file, " %u", *n);
				offset++;
			}
		}
		fprintf(file, "\n");
		face_offsets[e] = offset;
	}
	
	fprintf(file, "        </DataArray>\n");

	// Write cells faces offsets
	fprintf(file, "        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n");
	for (size_t e = 0; e < mesh->n_cells; ++e) {
		// number_of faces face_number_of
		fprintf(file, "          %u\n", face_offsets[e]);
	}
	free(face_offsets);
	fprintf(file, "        </DataArray>\n");

	fprintf(file, "      </Cells>\n");

	fprintf(file, "    </Piece>\n");
	fprintf(file, "  </UnstructuredGrid>\n");
	fprintf(file, "</VTKFile>\n");


	fclose(file);

    return 0;
}

