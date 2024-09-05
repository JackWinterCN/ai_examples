BUILD_DIR="../build"
PACKAGE_OUTPUT_DIR="../output"
MODEL_PARAM_DIR="../model_param"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "mkdir ${BUILD_DIR}"
    mkdir ${BUILD_DIR}
fi

if [ ! -d "${PACKAGE_OUTPUT_DIR}" ]; then
    echo "mkdir ${PACKAGE_OUTPUT_DIR}"
    mkdir ${PACKAGE_OUTPUT_DIR}
fi

cd ${BUILD_DIR}
# rm ./* -rf
cmake ..
cmake --build .
cp ./*_minist ${PACKAGE_OUTPUT_DIR}
cp ${MODEL_PARAM_DIR}/* ${PACKAGE_OUTPUT_DIR} -r