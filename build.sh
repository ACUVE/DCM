set -ve

OUTDIRECTORY="linux"
CXXFLAGS="-Wall -O3 -std=c++1z -I ../../library/fadeRelease/include_fade2d"
LIBS="-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -pthread"
FADE2DLIBS="-L../../library/fadeRelease/lib_fedora24_x86_64 -Wl,-rpath=../../library/fadeRelease/lib_fedora24_x86_64 -lfade2d"

cd `dirname $0`

mkdir -p "${OUTDIRECTORY}"
cd "${OUTDIRECTORY}"

for f in ../DCM/*.cpp
do
    g++ ${CXXFLAGS} -c "$f"
done

g++ ${FADE2DLIBS} ${LIBS} *.o -o DCM
