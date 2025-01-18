export GRPC_SERVER_PATH="../server/grpc_server"
export PYTHON_PKG="grpc_server"

python3 -m grpc_tools.protoc -I. --python_out=$GRPC_SERVER_PATH --pyi_out=$GRPC_SERVER_PATH --grpc_python_out=$GRPC_SERVER_PATH service.proto

# Use sed to replace the import statement with the correct package
sed -i "s|import service_pb2 as service__pb2|import ${PYTHON_PKG}.service_pb2 as service__pb2|g" $GRPC_SERVER_PATH/service_pb2_grpc.py