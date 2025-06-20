function run_docker_wk() {
   
    # 镜像名称和版本
    IMAGE_NAME="zico_swap_image:v1.0"

    # 容器名称（可选）
    CONTAINER_NAME="dev_aznas_container_search"


    # 检查 Docker 镜像是否存在
    if ! sudo docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        echo "Error: Docker image '$IMAGE_NAME' does not exist."
        return 1
    fi

    # 运行 Docker 容器
    sudo docker run -dit \
        --gpus all --shm-size=8g\
        -v "/home/ubuntu/scratch:/home/ubuntu/scratch" \
        -w /home/ubuntu/scratch \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME" \
        tail -f /dev/null
}

# 调用函数（可选，如果需要直接执行）
run_docker_wk
