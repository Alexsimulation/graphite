
CC = nvcc

.SUFFIXES: .cu .exe

APP = graphite.exe
TARGET = build/$(APP)
SRC_MAIN = apps/main.cpp


VS_MSVCRT_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\lib\x64\uwp"
GLFW_LIB_PATH = C:\glfw-3.3.8.bin.WIN64\lib-vc2019
GLEW_LIB_PATH = C:\glew-2.1.0\lib\Release\x64

GLFW_INCLUDE_PATH = C:\glfw-3.3.8.bin.WIN64\include
GLEW_INCLUDE_PATH = C:\glew-2.1.0\include

OPENGL_INCLUDES = -I$(GLFW_INCLUDE_PATH) -I$(GLEW_INCLUDE_PATH)
OPENGL_LIBS = -lmsvcrt -luser32 -lgdi32 -lucrt -lshell32 -llibcmt \
	-L$(GLEW_LIB_PATH) \
	-L$(GLFW_LIB_PATH) \
	-L$(VS_MSVCRT_PATH) \
	-lglew32s -lopengl32 -lglu32 -lglfw3


INCLUDE = -I./include $(OPENGL_INCLUDES)
LIBS = $(OPENGL_LIBS)
OPTIM = -O3 --use_fast_math

# OPTIONS = -DFVM_OUTPUT_GRAD_LIM

SRCS = \
	$(SRC_MAIN) \
	./src/flexarray.cu \
	./src/mesh.cu \
	./src/meshio.cu \
	./src/fvmfields.cu \
	./src/fvmfieldsio.cu \
	./src/fvmops.cu \
	./src/models/euler.cu \
	./src/models/model.cu \
	./src/render/window.cu

OBJS = \
	./build/obj/main.obj \
	./build/obj/flexarray.obj \
	./build/obj/mesh.obj \
	./build/obj/meshio.obj \
	./build/obj/fvmfields.obj \
	./build/obj/fvmfieldsio.obj \
	./build/obj/fvmops.obj \
	./build/obj/euler.obj \
	./build/obj/model.obj \
	./build/obj/window.obj

CFLAGS = $(INCLUDE) $(OPTIM) $(OPTIONS)
LFLAGS = $(LIBS) $(OPTIM)

{./src/}.cu{./build/obj/}.obj:
	$(CC) $(CFLAGS) -c $< -o $@

{./apps/}.cu{./build/obj/}.obj:
	$(CC) $(CFLAGS) -c $< -o $@

{./src/models/}.cu{./build/obj/}.obj:
	$(CC) $(CFLAGS) -c $< -o $@

{./src/render/}.cu{./build/obj/}.obj:
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o $(TARGET)
	cp $(TARGET) $(APP)

clean:
	rm -f build/*.exe
	rm -f build/*.exp
	rm -f build/*.lib
	rm -f build/obj/*
	rm -f $(APP)
	rm -rf log*


