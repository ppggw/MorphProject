TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += qt

SOURCES += \
        detector.cpp \
        main.cpp \
        object.cpp \
        observer.cpp \

HEADERS += \
    INIReader.h \
    cuda_header.h \
    detector.h \
    object.h \
    observer.h


DISTFILES += \
    param_mine.ini

INCLUDEPATH += /usr/include/opencv4

LIBS += -L/usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn


INCLUDEPATH += /usr/local/cuda-10.2/include/
INCLUDEPATH += /usr/local/cuda-10.2/samples/common/inc/

LIBS += -L/usr/local/cuda-10.2/lib64/ -lcuda -lcudart -lcufft

CUDASOURCES = cuda_calc.cu

cu.output = ${QMAKE_FILE_BASE}.o
cu.commands = /usr/local/cuda/bin/nvcc -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cu.input = CUDASOURCES
cu.CONFIG += no_link
cu.variable_out = OBJECTS

QMAKE_EXTRA_COMPILERS += cu
